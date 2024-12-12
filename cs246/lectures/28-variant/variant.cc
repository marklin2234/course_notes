import <variant>;
import <iostream>;
import <vector>;

using std::cout;
using std::endl;

class Turtle {
 public:
  void stealShell() {
    cout << "You stole my shell!\n";
  }
};

class Bullet {
 public:
  void deflect() {
    cout << "I've been deflected!\n";
  }
};

using Enemy = std::variant<Turtle, Bullet>;

int main() {
  std::vector<Enemy> v {Turtle{}, Bullet{}, Bullet{}, Turtle{}, Bullet{}};

  for(auto &e: v) {
    if (holds_alternative<Turtle>(e)) {
      get<Turtle>(e).stealShell();
    }
    else get<Bullet>(e).deflect();
  }
}

