import <iostream>;
import <ranges>;
import <vector>;

using std::ranges::views::transform;
using std::ranges::views::filter;

auto odd = [](int n) { return n % 2 != 0; };
auto sqr = [](int n) { return n * n; };

int main() {
  std::vector v {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto x = v | filter(odd) | transform(sqr);

  for (auto n : x) { std::cout << n << ' '; }
  std::cout << std::endl;
}
