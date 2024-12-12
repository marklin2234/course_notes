#include "rectangle.h"
#include <string>

Rectangle::Rectangle()
    : point{Point{0, 0}}, height{0}, width{0}, colour{Colour::Red} {};

Rectangle::Rectangle(Point upperLeft, int height, int width, Colour colour)
    : point{upperLeft}, height{height}, width{width}, colour{colour} {}

// Precondition: getPoint().getX() + x >= 0 and
//    getPoint().getY() + y >= 0
// Postcondition: Point = {getPoint().getX() + x, getPoint().getY() + y}
void Rectangle::translate(int x, int y) {
  if (this->point.getX() + x < 0 || this->point.getY() + y < 0) {
    return;
  }

  this->point = this->point + Point(x, y);
}

// Precondition: heightFactor > 0 and widthFactor > 0
// Scales the Rectangle's dimensions by the specified amounts.
// Postcondition: new height = old height * heightFactor
//    and new width = old width * widthFactor
void Rectangle::scale(float heightFactor, float widthFactor) {
  if (heightFactor <= 0 || widthFactor <= 0) {
    return;
  }

  this->height *= heightFactor;
  this->width *= widthFactor;
}

// Changes the colour to the new colour.
void Rectangle::change(Colour newColour) { this->colour = newColour; }

Colour Rectangle::getColour() const { return this->colour; }
Point Rectangle::getPoint() const { return this->point; }
int Rectangle::getWidth() const { return this->width; }
int Rectangle::getHeight() const { return this->height; }

Colour translateColour(char c) {
  switch (c) {
  case 'r':
    return Colour::Red;
  case 'g':
    return Colour::Green;
  case 'b':
    return Colour::Blue;
  case 'y':
    return Colour::Yellow;
  case 'o':
    return Colour::Orange;
  case 'a':
    return Colour::Black;
  case 'w':
    return Colour::White;
  }
  return Colour::Red;
} // translate

std::string colourToString(Colour c) {
  switch (c) {
  case Colour::Red:
    return "Red";
  case Colour::Green:
    return "Green";
  case Colour::Blue:
    return "Blue";
  case Colour::Yellow:
    return "Yellow";
  case Colour::Orange:
    return "Orange";
  case Colour::Black:
    return "Black";
  case Colour::White:
    return "White";
  }
  return "Red";
}

// Reads in a Rectangle from the specified input stream. Input format consists
// of: colour x-coordinate y-coordinate height width
// where colour is a character such that 'r' => Colour::Red, 'g' =>
// Colour::Green,
//    'b' => Colour::Blue, 'o' => Colour::Orange, 'y' = Colour::Yellow,
//    'a' => Colour::Black and 'w' => Colour::White and the other 4 values are
//    integers.
std::istream &operator>>(std::istream &in, Rectangle &rectangle) {
  char c;
  int x, y, h, w;

  in >> c >> x >> y >> h >> w;

  Rectangle rect = Rectangle(Point(x, y), h, w, translateColour(c));
  rectangle = rect;
  return in;
}

// Outputs a Rectangle to the specified output stream. Output format is:
// [colour (x,y) heightxwidth]
std::ostream &operator<<(std::ostream &out, const Rectangle &rectangle) {
  out << "[" << colourToString(rectangle.getColour()) << " "
      << rectangle.getPoint() << " " << rectangle.getHeight() << "x"
      << rectangle.getWidth() << "]";

  return out;
}
