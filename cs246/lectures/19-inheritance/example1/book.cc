export module book;
import <string>;

export class Book {
  std::string title, author;
  int numPages;
 public:
  Book(const std::string &title, const std::string &author, int numPages);
  std::string getTitle() const;
  std::string getAuthor() const;
  int getNumPages() const;
  bool isItHeavy() const;
};
