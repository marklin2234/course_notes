module comic;
import <string>;

using namespace std;

Comic::Comic(const string &title, const string &author, int numPages, const string &hero):
  Book{title, author, numPages}, hero{hero} {}

bool Comic::isItHeavy() const { return getNumPages() > 30; }
string Comic::getHero() const { return hero; }

// My favourite comics are Superman comics.
bool Comic::favourite() const { return hero == "Superman"; }
