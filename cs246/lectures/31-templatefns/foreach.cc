import <iostream>;

template<typename Iter, typename Fn>
void foreach(Iter start, Iter finish, Fn f) {
  while(start != finish) {
    f(*start);
    ++start;
  }
}

int main() {
  int a[] = {1, 2, 3, 4, 5};
  foreach(a, a+5, [](int n) { std::cout << n << ' '; });
  std::cout << std::endl;
}
