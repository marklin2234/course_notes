import <iostream>;
using namespace std;

struct Student {
  int assigns, mt, final;

  mutable int numMethodCalls;     // Track number of method calls

  float computeGrade() const {
    ++numMethodCalls;
    return assigns * 0.4 + mt * 0.2 + final * 0.4;
  }
  explicit Student(int assigns = 0, int mt = 0, int final = 0):
     assigns(assigns), mt(mt), final(final), numMethodCalls(0) {}

  Student(const Student &other): assigns(other.assigns),
                                 mt(other.mt),
                                 final(other.final),
                                 numMethodCalls(0) {}
};

int main () {
  Student s(60, 70, 80);
  const Student s2 = s;
  cout << s.computeGrade() << endl;
  cout << s2.computeGrade() << endl;
  cout << "# method calls for s2: " << s2.numMethodCalls << endl;
}
