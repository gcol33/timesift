#include "ts_tree.h"

#include <cmath>
#include <memory>
#include <string>

#include "ts_core.h"

// Every sum, product and quotient here is rpart's, operation for operation, and a fused
// multiply-add rounds once where rpart rounds twice. The difference is a unit in the last place,
// and a unit in the last place is enough to turn a tie between two splits, or between a split and
// the complexity threshold, the other way. Contraction is off so the tree is the same on every
// machine: clang contracts inside an expression by default, and GCC across expressions wherever the
// target has the instruction.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace timesift {
namespace {

// rpart's two directions a continuous split sends the values below its threshold.
constexpr int kLeft = -1;
constexpr int kRight = 1;

struct Node {
  double risk = 0.0;
  double complexity = 0.0;
  double sum_wt = 0.0;
  int num_obs = 0;
  // The node's estimate: for the Gini rule the class it predicts (1 or 2), the weighted count of
  // each class and the probability of a 1; for the sum of squares the mean alone.
  double est[4] = {0.0, 0.0, 0.0, 0.0};
  bool split = false;
  int var = -1;
  double spoint = 0.0;
  int direction = kLeft;
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
};

// rpart's quicksort of one column carrying the observation indices along. Its order among tied
// values is not a stable one, and the order observations are summed in is the order they sit in
// here, so the sort is rpart's own rather than std::sort: two sums of the same numbers in two
// orders can differ in the last place and turn a tie between two columns the other way.
void rpart_sort(int start, int stop, double* x, int* cvec) {
  while (start < stop) {
    if ((stop - start) < 11) {
      for (int i = start + 1; i <= stop; i++) {
        const double temp = x[i];
        const int tempd = cvec[i];
        int j = i - 1;
        while (j >= start && x[j] > temp) {
          x[j + 1] = x[j];
          cvec[j + 1] = cvec[j];
          j--;
        }
        x[j + 1] = temp;
        cvec[j + 1] = tempd;
      }
      return;
    }
    int i = start;
    int j = stop;
    const int k = (start + stop) / 2;
    double median = x[k];
    if (x[i] >= x[k]) {
      if (x[j] > x[k]) {
        median = (x[i] > x[j]) ? x[j] : x[i];
      }
    } else if (x[j] < x[k]) {
      median = (x[i] > x[j]) ? x[i] : x[j];
    }
    while (i < j) {
      while (x[i] < median) i++;
      while (x[j] > median) j--;
      if (i < j) {
        if (x[i] > x[j]) {
          const double temp = x[i];
          x[i] = x[j];
          x[j] = temp;
          const int tempd = cvec[i];
          cvec[i] = cvec[j];
          cvec[j] = tempd;
        }
        i++;
        j--;
      }
    }
    while (x[i] >= median && i > start) i--;
    while (x[j] <= median && j < stop) j++;
    if ((i - start) < (stop - j)) {
      if ((i - start) > 0) rpart_sort(start, i, x, cvec);
      start = j;
    } else {
      if ((stop - j) > 0) rpart_sort(j, stop, x, cvec);
      stop = i;
    }
  }
}

// One row of the complexity table, linked the way rpart links them so the table is filled in the
// same order and with the same sums.
struct CpRow {
  double cp = 0.0;
  double risk = 0.0;
  double xrisk = 0.0;
  double xstd = 0.0;
  int nsplit = 0;
  int forward = -1;
  int back = -1;
};

class Grower {
 public:
  Grower(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
         bool gini, const TreeSpec& spec)
      : x_(x), y_(y), w_(w), n_(static_cast<int>(n)), p_(static_cast<int>(p)), gini_(gini),
        min_split_(spec.min_split), min_node_(spec.min_leaf),
        maxnode_(static_cast<long long>(std::pow(2.0, spec.max_depth)) - 1),
        complexity_(spec.cp), sorts_(n * p), side_(n), tempvec_(n), xtemp_(n), ytemp_(n),
        wtemp_(n) {
    std::vector<double> column(n);
    for (int v = 0; v < p_; ++v) {
      int* index = sorts_.data() + static_cast<std::size_t>(v) * n;
      for (int k = 0; k < n_; ++k) {
        index[k] = k;
        column[k] = x_[k + static_cast<std::size_t>(v) * n];
      }
      rpart_sort(0, n_ - 1, column.data(), index);
    }
  }

  // The tree on every observation, its complexity table, and that table's cross-validated error
  // where folds are given.
  std::unique_ptr<Node> grow(std::vector<CpRow>& table, int& tail, const std::int32_t* fold,
                             int n_fold) {
    std::vector<int> saved;
    if (n_fold > 1) saved = sorts_;
    auto root = std::make_unique<Node>();
    double twt = 0.0;
    for (int i = 0; i < n_; ++i) {
      ytemp_[i] = y_[i];
      wtemp_[i] = w_[i];
      twt += w_[i];
    }
    root->num_obs = n_;
    root->sum_wt = twt;
    evaluate(n_, root.get());
    root->complexity = root->risk;
    alpha_ = complexity_ * root->risk;
    double sumrisk = 0.0;
    partition(1, root.get(), &sumrisk, 0, n_);

    table.clear();
    CpRow head;
    head.cp = root->complexity;
    head.risk = root->risk;
    table.push_back(head);
    tail = 0;
    if (root->left) {
      make_cp_list(root.get(), root->complexity, table, tail);
      make_cp_table(root.get(), root->complexity, 0, table, tail);
      if (n_fold > 1) xval(n_fold, table, fold, saved);
    }
    return root;
  }

  double alpha() const { return alpha_; }

 private:
  // The node's estimate and its risk from the observations in `ytemp_` and `wtemp_`: rpart's
  // `ginidev` with the priors the data's own class shares, which reduces every prior to one, and
  // its `anovass`.
  void evaluate(int n, Node* me) {
    if (gini_) {
      double freq0 = 0.0;
      double freq1 = 0.0;
      double temp = 0.0;
      for (int i = 0; i < n; ++i) {
        if (ytemp_[i] == 0.0) {
          freq0 += wtemp_[i];
        } else {
          freq1 += wtemp_[i];
        }
        temp += wtemp_[i];
      }
      // Predicting class 1 misclassifies the weight of class 2 and the other way round; the first
      // class wins a tie.
      int max = 0;
      double dev = freq1;
      if (freq0 < dev) {
        max = 1;
        dev = freq0;
      }
      me->est[0] = max + 1;
      me->est[1] = freq0;
      me->est[2] = freq1;
      me->est[3] = temp;
      me->risk = dev;
    } else {
      double temp = 0.0;
      double twt = 0.0;
      for (int i = 0; i < n; ++i) {
        temp += ytemp_[i] * wtemp_[i];
        twt += wtemp_[i];
      }
      const double mean = temp / twt;
      double ss = 0.0;
      for (int i = 0; i < n; ++i) {
        const double d = ytemp_[i] - mean;
        ss += d * d * wtemp_[i];
      }
      me->est[0] = mean;
      me->risk = ss;
    }
  }

  // The loss of predicting `est` for an observation of response `y`: a misclassification for the
  // Gini rule, the squared error for the sum of squares.
  double error(double y, const double* est) const {
    if (gini_) return (static_cast<int>(y) + 1 == static_cast<int>(est[0])) ? 0.0 : 1.0;
    const double d = y - est[0];
    return d * d;
  }

  // The best split of one column's observations in `xtemp_`, sorted, with their responses and
  // weights beside them: rpart's `gini` and `anova` on a continuous predictor.
  void choose(int n, double my_risk, double* improve, double* split, int* direction) {
    const int edge = min_node_;
    const double* x = xtemp_.data();
    const double* y = ytemp_.data();
    const double* wt = wtemp_.data();
    if (gini_) {
      double left[2] = {0.0, 0.0};
      double right[2] = {0.0, 0.0};
      double lwt = 0.0;
      double rwt = 0.0;
      int rtot = 0;
      int ltot = 0;
      for (int i = 0; i < n; ++i) {
        const int j = y[i] == 0.0 ? 0 : 1;
        rwt += wt[i];
        right[j] += wt[i];
        rtot++;
      }
      double total_ss = 0.0;
      for (int i = 0; i < 2; ++i) {
        const double temp = right[i] / rwt;
        total_ss += rwt * (temp * (1.0 - temp));
      }
      double best = total_ss;
      int where = 0;
      int dir = kLeft;
      for (int i = 0; rtot > edge; i++) {
        const int j = y[i] == 0.0 ? 0 : 1;
        rwt -= wt[i];
        lwt += wt[i];
        rtot--;
        ltot++;
        right[j] -= wt[i];
        left[j] += wt[i];
        if (ltot >= edge && x[i + 1] != x[i]) {
          double temp = 0.0;
          double lmean = 0.0;
          double rmean = 0.0;
          for (int c = 0; c < 2; ++c) {
            double pr = left[c] / lwt;
            temp += lwt * (pr * (1.0 - pr));
            lmean += pr * c;
            pr = right[c] / rwt;
            temp += rwt * (pr * (1.0 - pr));
            rmean += pr * c;
          }
          if (temp < best) {
            best = temp;
            where = i;
            dir = lmean < rmean ? kLeft : kRight;
          }
        }
      }
      *improve = total_ss - best;
      if (*improve > 0) {
        *direction = dir;
        *split = (x[where] + x[where + 1]) / 2;
      }
      return;
    }
    double right_sum = 0.0;
    double right_wt = 0.0;
    int right_n = n;
    for (int i = 0; i < n; ++i) {
      right_sum += y[i] * wt[i];
      right_wt += wt[i];
    }
    const double grandmean = right_sum / right_wt;
    double left_sum = 0.0;
    double left_wt = 0.0;
    int left_n = 0;
    right_sum = 0.0;
    double best = 0.0;
    int where = 0;
    int dir = kLeft;
    for (int i = 0; right_n > edge; i++) {
      left_wt += wt[i];
      right_wt -= wt[i];
      left_n++;
      right_n--;
      const double temp = (y[i] - grandmean) * wt[i];
      left_sum += temp;
      right_sum -= temp;
      if (x[i + 1] != x[i] && left_n >= edge) {
        const double t = left_sum * left_sum / left_wt + right_sum * right_sum / right_wt;
        if (t > best) {
          best = t;
          where = i;
          dir = left_sum < right_sum ? kLeft : kRight;
        }
      }
    }
    *improve = best / my_risk;
    if (best > 0) {
      *direction = dir;
      *split = (x[where] + x[where + 1]) / 2;
    }
  }

  // rpart's `bsplit`: every column in turn, the first to reach the largest improvement kept.
  void best_split(Node* me, int n1, int n2) {
    me->split = false;
    double best = 0.0;
    for (int v = 0; v < p_; ++v) {
      const int* index = sorts_.data() + static_cast<std::size_t>(v) * n_;
      const double* column = x_ + static_cast<std::size_t>(v) * n_;
      int k = 0;
      for (int j = n1; j < n2; ++j) {
        const int kk = index[j];
        if (w_[kk] > 0) {
          xtemp_[k] = column[kk];
          ytemp_[k] = y_[kk];
          wtemp_[k] = w_[kk];
          k++;
        }
      }
      if (k == 0 || xtemp_[0] == xtemp_[k - 1]) continue;
      double improve = 0.0;
      double split = 0.0;
      int direction = kLeft;
      choose(k, me->risk, &improve, &split, &direction);
      // rpart's guard against a rounding error posing as an improvement, read against the
      // largest improvement the whole fit has seen so far.
      if (improve > iscale_) iscale_ = improve;
      if (improve > iscale_ * 1e-10) {
        if (!me->split || improve > best) {
          me->split = true;
          best = improve;
          me->var = v;
          me->spoint = split;
          me->direction = direction;
        }
      }
    }
  }

  // rpart's `nodesplit` without missing values: each observation sent left or right, and every
  // column's sorted index reordered to the left child's observations then the right child's, in
  // the order they held.
  void split_node(const Node* me, int n1, int n2, int* nleft, int* nright) {
    const double* column = x_ + static_cast<std::size_t>(me->var) * n_;
    const int* pindex = sorts_.data() + static_cast<std::size_t>(me->var) * n_;
    int nl = 0;
    int nr = 0;
    for (int i = n1; i < n2; ++i) {
      const int j = pindex[i];
      const int k = (column[j] < me->spoint) ? me->direction : -me->direction;
      if (k == kLeft) {
        side_[j] = 1;
        nl++;
      } else {
        side_[j] = 2;
        nr++;
      }
    }
    for (int v = 0; v < p_; ++v) {
      int* sindex = sorts_.data() + static_cast<std::size_t>(v) * n_;
      int i1 = n1;
      int i2 = n1 + nl;
      for (int i = n1; i < n2; ++i) {
        const int j = sindex[i];
        if (side_[j] == 1) {
          sindex[i1++] = j;
        } else {
          tempvec_[i2++] = j;
        }
      }
      for (int i = n1 + nl; i < n2; ++i) sindex[i] = tempvec_[i];
    }
    *nleft = nl;
    *nright = nr;
  }

  // rpart's `partition`, returning the splits kept below `me` and the risk of its subtree.
  int partition(long long nodenum, Node* me, double* sumrisk, int n1, int n2) {
    const int n = n2 - n1;
    double tempcp;
    if (nodenum > 1) {
      const int* first = sorts_.data();
      double twt = 0.0;
      int k = 0;
      for (int i = n1; i < n2; ++i) {
        const int j = first[i];
        wtemp_[k] = w_[j];
        ytemp_[k] = y_[j];
        twt += w_[j];
        k++;
      }
      evaluate(n, me);
      me->num_obs = n;
      me->sum_wt = twt;
      tempcp = me->risk;
      if (tempcp > me->complexity) tempcp = me->complexity;
    } else {
      tempcp = me->risk;
    }

    if (me->num_obs < min_split_ || tempcp <= alpha_ || nodenum > maxnode_) {
      me->complexity = alpha_;
      *sumrisk = me->risk;
      me->split = false;
      return 0;
    }

    best_split(me, n1, n2);
    if (!me->split) {
      me->complexity = alpha_;
      *sumrisk = me->risk;
      return 0;
    }

    int nleft = 0;
    int nright = 0;
    split_node(me, n1, n2, &nleft, &nright);

    me->left = std::make_unique<Node>();
    me->left->complexity = tempcp - alpha_;
    double left_risk = 0.0;
    int left_split = partition(2 * nodenum, me->left.get(), &left_risk, n1, n1 + nleft);

    tempcp = (me->risk - left_risk) / (left_split + 1);
    const double tempcp2 = me->risk - me->left->risk;
    if (tempcp < tempcp2) tempcp = tempcp2;
    if (tempcp > me->complexity) tempcp = me->complexity;

    me->right = std::make_unique<Node>();
    me->right->complexity = tempcp - alpha_;
    double right_risk = 0.0;
    int right_split = partition(1 + 2 * nodenum, me->right.get(), &right_risk, n1 + nleft,
                                n1 + nleft + nright);

    tempcp = (me->risk - (left_risk + right_risk)) / (left_split + right_split + 1);
    if (me->right->complexity > me->left->complexity) {
      if (tempcp > me->left->complexity) {
        left_risk = me->left->risk;
        left_split = 0;
        tempcp = (me->risk - (left_risk + right_risk)) / (left_split + right_split + 1);
        if (tempcp > me->right->complexity) {
          right_risk = me->right->risk;
          right_split = 0;
        }
      }
    } else if (tempcp > me->right->complexity) {
      right_split = 0;
      right_risk = me->right->risk;
      tempcp = (me->risk - (left_risk + right_risk)) / (left_split + right_split + 1);
      if (tempcp > me->left->complexity) {
        left_risk = me->left->risk;
        left_split = 0;
      }
    }
    me->complexity = (me->risk - (left_risk + right_risk)) / (left_split + right_split + 1);

    if (me->complexity <= alpha_) {
      me->left.reset();
      me->right.reset();
      me->split = false;
      *sumrisk = me->risk;
      return 0;
    }
    *sumrisk = left_risk + right_risk;
    return left_split + right_split + 1;
  }

  void make_cp_list(Node* me, double parent, std::vector<CpRow>& table, int& tail) {
    if (me->complexity > parent) me->complexity = parent;
    double me_cp = me->complexity;
    if (me_cp < alpha_) me_cp = alpha_;
    if (me->left) {
      make_cp_list(me->left.get(), me_cp, table, tail);
      make_cp_list(me->right.get(), me_cp, table, tail);
    }
    if (me_cp < parent) {
      int temp = -1;
      for (int c = 0; c != -1; c = table[c].forward) {
        if (me_cp == table[c].cp) return;
        if (me_cp > table[c].cp) break;
        temp = c;
      }
      CpRow row;
      row.cp = me_cp;
      row.back = temp;
      row.forward = table[temp].forward;
      const int at = static_cast<int>(table.size());
      table.push_back(row);
      if (table[at].forward != -1) {
        table[table[at].forward].back = at;
      } else {
        tail = at;
      }
      table[temp].forward = at;
    }
  }

  int make_cp_table(const Node* me, double parent, int nsplit, std::vector<CpRow>& table,
                    int tail) {
    int c;
    if (me->left) {
      make_cp_table(me->left.get(), me->complexity, 0, table, tail);
      c = make_cp_table(me->right.get(), me->complexity, nsplit + 1, table, tail);
    } else {
      c = tail;
    }
    while (table[c].cp < parent) {
      table[c].risk += me->risk;
      table[c].nsplit += nsplit;
      c = table[c].back;
    }
    return c;
  }

  static void fix_cp(Node* me, double parent_cp) {
    if (me->complexity > parent_cp) me->complexity = parent_cp;
    if (me->left) {
      fix_cp(me->left.get(), me->complexity);
      fix_cp(me->right.get(), me->complexity);
    }
  }

  // The losses of one held-out observation at every complexity of the table: rpart's `rundown`,
  // descending while the complexity asked for is below the node's.
  void rundown(const Node* tree, int obs, const std::vector<double>& cp,
               std::vector<double>& loss) const {
    for (std::size_t i = 0; i < cp.size(); ++i) {
      while (cp[i] < tree->complexity && tree->left) {
        const double v = x_[obs + static_cast<std::size_t>(tree->var) * n_];
        const int dir = (v < tree->spoint) ? tree->direction : -tree->direction;
        tree = (dir == kLeft) ? tree->left.get() : tree->right.get();
      }
      loss[i] = error(y_[obs], tree->est);
    }
  }

  // rpart's `xval`: a tree grown on each fold's complement under the complexity rescaled to its
  // weight, and each held-out observation's loss read at every row of the table.
  void xval(int n_xval, std::vector<CpRow>& table, const std::int32_t* fold,
            const std::vector<int>& saved) {
    const double alphasave = alpha_;
    std::vector<int> order;
    for (int c = 0; c != -1; c = table[c].forward) order.push_back(c);
    const std::size_t num = order.size();
    std::vector<double> cp(num);
    cp[0] = 10 * table[order[0]].cp;
    for (std::size_t i = 1; i < num; ++i) {
      cp[i] = std::sqrt(table[order[i - 1]].cp * table[order[i]].cp);
    }
    double total_wt = 0.0;
    for (int i = 0; i < n_; ++i) total_wt += w_[i];
    double old_wt = total_wt;
    std::vector<double> loss(num);

    for (int group = 0; group < n_xval; ++group) {
      int k = 0;
      for (int v = 0; v < p_; ++v) {
        k = 0;
        const int* from = saved.data() + static_cast<std::size_t>(v) * n_;
        int* to = sorts_.data() + static_cast<std::size_t>(v) * n_;
        for (int i = 0; i < n_; ++i) {
          if (fold[from[i]] != group) to[k++] = from[i];
        }
      }
      int last = k;
      k = 0;
      double temp = 0.0;
      for (int i = 0; i < n_; ++i) {
        if (fold[i] == group) {
          sorts_[last++] = i;
        } else {
          ytemp_[k] = y_[i];
          wtemp_[k] = w_[i];
          temp += w_[i];
          k++;
        }
      }
      for (std::size_t j = 0; j < num; ++j) cp[j] *= temp / old_wt;
      alpha_ *= temp / old_wt;
      old_wt = temp;

      Node xtree;
      xtree.num_obs = k;
      evaluate(k, &xtree);
      xtree.complexity = xtree.risk;
      double sumrisk = 0.0;
      partition(1, &xtree, &sumrisk, 0, k);
      fix_cp(&xtree, xtree.complexity);

      for (int i = k; i < n_; ++i) {
        const int j = sorts_[i];
        rundown(&xtree, j, cp, loss);
        for (std::size_t c = 0; c < num; ++c) {
          table[order[c]].xrisk += loss[c] * w_[j];
          table[order[c]].xstd += loss[c] * loss[c] * w_[j];
        }
      }
    }
    for (std::size_t c = 0; c < num; ++c) {
      CpRow& row = table[order[c]];
      row.xstd = std::sqrt(row.xstd - row.xrisk * row.xrisk / total_wt);
    }
    alpha_ = alphasave;
  }

  const double* x_;
  const double* y_;
  const double* w_;
  int n_;
  int p_;
  bool gini_;
  int min_split_;
  int min_node_;
  long long maxnode_;
  double complexity_;
  double alpha_ = 0.0;
  double iscale_ = 0.0;
  std::vector<int> sorts_;
  std::vector<std::int8_t> side_;
  std::vector<int> tempvec_;
  std::vector<double> xtemp_;
  std::vector<double> ytemp_;
  std::vector<double> wtemp_;
};

void check_finite(const double* v, std::size_t n, const char* what) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      throw Error(std::string("a tree is fitted on finite values, and the ") + what +
                  " holds one that is not, at position " + std::to_string(i + 1) + ".");
    }
  }
}

// The fitted nodes in rpart's frame order, a node's children after it where it keeps them.
void flatten(const Node* me, long long number, double scale, double alpha, bool gini, Tree& out) {
  const std::size_t at = out.number.size();
  out.number.push_back(static_cast<std::int32_t>(number));
  out.n.push_back(me->num_obs);
  out.weight.push_back(me->sum_wt);
  out.risk.push_back(me->risk);
  out.complexity.push_back(me->complexity * scale);
  if (gini) {
    const double total = me->est[1] + me->est[2];
    out.value.push_back(total > 0 ? me->est[2] / total : 0.0);
  } else {
    out.value.push_back(me->est[0]);
  }
  const bool kids = me->left && me->complexity > alpha;
  out.column.push_back(kids ? me->var : -1);
  out.threshold.push_back(kids ? me->spoint : 0.0);
  out.less_left.push_back(kids && me->direction == kLeft ? 1 : 0);
  out.left.push_back(-1);
  out.right.push_back(-1);
  if (!kids) return;
  out.left[at] = static_cast<std::int32_t>(out.number.size());
  flatten(me->left.get(), 2 * number, scale, alpha, gini, out);
  out.right[at] = static_cast<std::int32_t>(out.number.size());
  flatten(me->right.get(), 2 * number + 1, scale, alpha, gini, out);
}

void copy_node(const Tree& from, std::size_t i, double cp, Tree& out) {
  const std::size_t at = out.number.size();
  out.number.push_back(from.number[i]);
  out.n.push_back(from.n[i]);
  out.weight.push_back(from.weight[i]);
  out.risk.push_back(from.risk[i]);
  out.complexity.push_back(from.complexity[i]);
  out.value.push_back(from.value[i]);
  const bool kids = from.column[i] >= 0 && !(from.complexity[i] <= cp);
  out.column.push_back(kids ? from.column[i] : -1);
  out.threshold.push_back(kids ? from.threshold[i] : 0.0);
  out.less_left.push_back(kids ? from.less_left[i] : 0);
  out.left.push_back(-1);
  out.right.push_back(-1);
  if (!kids) return;
  out.left[at] = static_cast<std::int32_t>(out.number.size());
  copy_node(from, static_cast<std::size_t>(from.left[i]), cp, out);
  out.right[at] = static_cast<std::int32_t>(out.number.size());
  copy_node(from, static_cast<std::size_t>(from.right[i]), cp, out);
}

}  // namespace

Tree tree_fit(const double* x, const double* y, const double* w, std::size_t n, std::size_t p,
              Family family, const TreeSpec& spec, const std::int32_t* fold,
              std::int32_t n_fold) {
  if (n == 0 || p == 0) throw Error("a tree needs at least one observation and one column.");
  if (spec.min_split < 1 || spec.min_leaf < 1) {
    throw Error("a tree's `min_split` and `min_leaf` are at least one observation.");
  }
  if (!(spec.cp >= 0.0)) throw Error("a tree's `cp` is zero or more.");
  if (spec.max_depth < 0 || spec.max_depth > 30) {
    throw Error("a tree's `max_depth` is between 0 and 30, as rpart's is.");
  }
  check_finite(x, n * p, "design");
  check_finite(y, n, "response");
  check_finite(w, n, "weights");
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (w[i] < 0) throw Error("a tree's case weights are zero or more.");
    total += w[i];
    if (family == Family::binomial && y[i] != 0.0 && y[i] != 1.0) {
      throw Error("a tree on a binomial response reads 0 and 1 alone.");
    }
  }
  if (!(total > 0)) throw Error("a tree's case weights sum to more than zero.");
  if (fold != nullptr && n_fold > 1) {
    for (std::size_t i = 0; i < n; ++i) {
      if (fold[i] < 0 || fold[i] >= n_fold) {
        throw Error("a fold index lies between 0 and the number of folds less one.");
      }
    }
  }

  const bool gini = family == Family::binomial;
  Grower grower(x, y, w, n, p, gini, spec);
  std::vector<CpRow> table;
  int tail = 0;
  const bool cross = fold != nullptr && n_fold > 1;
  const std::unique_ptr<Node> root = grower.grow(table, tail, fold, cross ? n_fold : 0);

  Tree out;
  out.family = family;
  out.root_risk = root->risk;
  const double scale = root->risk > 0 ? 1 / root->risk : 1.0;
  flatten(root.get(), 1, scale, grower.alpha(), gini, out);
  const bool split = root->left != nullptr;
  for (int c = 0; c != -1; c = table[c].forward) {
    out.cp.push_back(table[c].cp * scale);
    out.nsplit.push_back(table[c].nsplit);
    out.rel_error.push_back(table[c].risk * scale);
    if (cross && split) {
      out.xerror.push_back(table[c].xrisk * scale);
      out.xstd.push_back(table[c].xstd * scale);
    }
  }
  return out;
}

Tree tree_prune(const Tree& tree, double cp) {
  Tree out;
  out.family = tree.family;
  out.root_risk = tree.root_risk;
  out.cp = tree.cp;
  out.nsplit = tree.nsplit;
  out.rel_error = tree.rel_error;
  out.xerror = tree.xerror;
  out.xstd = tree.xstd;
  if (!tree.number.empty()) copy_node(tree, 0, cp, out);
  return out;
}

void tree_predict(const Tree& tree, const double* x, std::size_t n, std::size_t p, double* out) {
  if (tree.number.empty()) throw Error("the tree holds no node to predict from.");
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t at = 0;
    while (tree.column[at] >= 0) {
      const std::size_t c = static_cast<std::size_t>(tree.column[at]);
      if (c >= p) throw Error("the design predicted on has fewer columns than the tree splits on.");
      const double v = x[i + c * n];
      if (!std::isfinite(v)) {
        throw Error("a tree predicts from finite values, and row " + std::to_string(i + 1) +
                    " holds one that is not in a column it splits on.");
      }
      const bool below = v < tree.threshold[at];
      const bool go_left = below == (tree.less_left[at] == 1);
      at = static_cast<std::size_t>(go_left ? tree.left[at] : tree.right[at]);
    }
    out[i] = tree.value[at];
  }
}

}  // namespace timesift
