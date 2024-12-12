import <iostream>;
import <string>;
import book;
import text;
import comic;

using namespace std;

int main() {
  Comic c("Robin Rescues Batman Twice", "Robin", 40, "Robin");

  Book b = c;

  cout << "The comic book: " << c.getTitle() << "; " << c.getAuthor()
       << "; " << (c.isItHeavy() ? "heavy" : "not heavy") << endl;

  cout << "The book: " << b.getTitle() << "; " << b.getAuthor()
       << "; " << (b.isItHeavy() ? "heavy" : "not heavy") << endl;

  cout << endl << "Through pointers: " << endl;

  Comic *pc = new Comic("Spiderman Unabridged", "Peter Parker", 100, "Spiderman");
  Book *pb = pc;

  cout << "The comic book ptr: " << pc->getTitle() << "; " << pc->getAuthor()
       << "; " << (pc->isItHeavy() ? "heavy" : "not heavy") << endl;

  cout << "The book ptr: " << pb->getTitle() << "; " << pb->getAuthor()
       << "; " << (pb->isItHeavy() ? "heavy" : "not heavy") << endl;

}
