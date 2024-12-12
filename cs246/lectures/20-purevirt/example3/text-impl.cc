module text;
import <iostream>;
import <string>;

using namespace std;

Text::Text(const string &title, const string &author, int numPages, const string &topic):
  AbstractBook{title, author, numPages}, topic{topic} {}

bool Text::isItHeavy() const { return getNumPages() > 500; }
string Text::getTopic() const { return topic; }

// My favourite textbooks are C++ books
bool Text::favourite() const { return topic == "C++"; }

Text::Text(const Text &other): AbstractBook{other}, topic{other.topic} {
  cout << "Running Text's copy ctor..." << endl;
}

Text &Text::operator=(const Text &rhs) {
  cout << "Text assignment operator running ... " << endl;

  if (this == &rhs) return *this;
  AbstractBook::operator=(rhs);
  topic = rhs.topic;
  return *this;
}
