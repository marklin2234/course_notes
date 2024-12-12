#include "point.h"

Point::Point(int x, int y) : x{x}, y{y} {}

// Returns the x-coordinate.
int Point::getX() const { return this->x; }

// Returns the y-coordinate.
int Point::getY() const { return this->y; }

// Precondition: getPoint().getX() + other.getX() >= 0 and
//               getPoint().getY() + other.getY() >= 0
// Returns: Point{this->x + other.x, this->y + other.y}
Point Point::operator+(const Point &other) {
  if (this->getX() + other.getX() < 0 || this->getY() + other.getY() < 0) {
    return *this;
  }

  return Point{this->x + other.getX(), this->y + other.getY()};
}

std::ostream &operator<<(std::ostream &out, const Point &point) {
  out << "(" << point.getX() << "," << point.getY() << ")";
  return out;
}
