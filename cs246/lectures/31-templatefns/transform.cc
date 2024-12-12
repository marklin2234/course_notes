import <iostream>;
import <vector>;
import <iterator>;
import <algorithm>;

using std::vector;

template<typename InIter, typename OutIter, typename Fn>
OutIter my_transform(InIter first, InIter last, OutIter result, Fn f) {
  while (first != last) {
    *result = f(*first);
    ++first;
    ++result;
  }
  return result;
}

class Plus {
  int m;
 public:
  Plus(int m): m{m} {}
  int operator()(int n) { return n + m; }
};

class IncreasingPlus {
  int m = 0;
 public:
  int operator()(int n) { return n + (m++); }
  void reset() { m = 0; }
};

int main() {
  vector v(5, 0);
  vector<int> w(v.size()), x(v.size());

  my_transform(v.begin(), v.end(), w.begin(), Plus{2});
  my_transform(v.begin(), v.end(), x.begin(), IncreasingPlus{});

  std::ostream_iterator<int> out { std::cout, ", " };
  std::copy(w.begin(), w.end(), out);
  std::cout << std::endl;
  std::copy(x.begin(), x.end(), out);
  std::cout << std::endl;

  std::copy(w.begin(), w.end(), std::back_inserter(x));
  std::copy(x.begin(), x.end(), out);
  std::cout << std::endl;

}
