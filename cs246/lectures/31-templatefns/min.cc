import <iostream>;

template <typename T> T min (T x, T y) {
  return x < y ? x : y;
}

int main() {
  int x = 1, y = 2;
  std::cout << min(x, y) << std::endl;
}

