#include "canvas.h"

Canvas::Canvas()
    : height{0}, width{0}, n_Rect{0}, c_Rect{DEFAULT_RECT_SIZE},
      rectangles{new Rectangle[c_Rect]} {}
Canvas::Canvas(const Canvas &other)
    : height{other.height}, width{other.width}, n_Rect{other.n_Rect},
      c_Rect{other.c_Rect} {
  rectangles = new Rectangle[c_Rect];
  for (int i = 0; i < n_Rect; i++) {
    rectangles[i] = Rectangle(other.rectangles[i]);
  }
}
Canvas::Canvas(Canvas &&other)
    : height{other.height}, width{other.width}, n_Rect{other.n_Rect},
      c_Rect{other.c_Rect} {
  this->rectangles = other.rectangles;
  other.rectangles = nullptr;
}

Canvas::~Canvas() { delete[] this->rectangles; }

Canvas &Canvas::operator=(const Canvas &other) {
  if (&other == this) {
    return *this;
  }

  delete[] rectangles;
  height = other.height;
  width = other.width;
  c_Rect = other.c_Rect;
  n_Rect = other.n_Rect;

  rectangles = new Rectangle[c_Rect];
  for (int i = 0; i < n_Rect; i++) {
    rectangles[i] = other.rectangles[i];
  }

  return *this;
}

Canvas &Canvas::operator=(Canvas &&other) {
  if (&other == this) {
    return *this;
  }

  delete[] rectangles;
  height = other.height;
  width = other.width;
  c_Rect = other.c_Rect;
  n_Rect = other.n_Rect;

  rectangles = new Rectangle[c_Rect];
  for (int i = 0; i < n_Rect; i++) {
    rectangles[i] = other.rectangles[i];
  }

  other.rectangles = nullptr;
  return *this;
}

// Adds the given Rectangle after already existing rectangles.
// Postcondition: The dimensions of the Canvas "stretch" to fit the Rectangle,
// if necessary, depending upon where the Rectangle's upper-left-hand corner
// is defined to be and its dimensions.
void Canvas::add(const Rectangle &rectangle) {
  rectangles[n_Rect++] = rectangle;

  if (n_Rect == c_Rect) {
    resizeRectangles(c_Rect * 2);
  }

  width = std::max(width, rectangle.getPoint().getX() + rectangle.getWidth());
  height =
      std::max(height, rectangle.getPoint().getY() + rectangle.getHeight());
}

// Returns the number of rectangles in the Canvas.
int Canvas::numRectangles() const { return n_Rect; }

// Returns the width of the Canvas.
int Canvas::getWidth() const { return width; }

// Returns the height of the Canvas.
int Canvas::getHeight() const { return height; }

// Precondition: 0 <= rectangleId < numRectangles()
// Returns array[rectangleId].
Rectangle Canvas::get(int rectangleId) const { return rectangles[rectangleId]; }

// Precondition: 0 <= rectangleId < numRectangles() and
//    array[rectangleId].getPoint().getX() + x >= 0 and
//    array[rectangleId].getPoint().getY() + y >= 0
// Postcondition: array[rectangleId].translate(x,y) and
//    Canvas dimensions may have been altered as a result
void Canvas::translate(int rectangleId, int x, int y) {
  if (rectangleId < 0 || rectangleId >= n_Rect) {
    return;
  }

  rectangles[rectangleId].translate(x, y);
  updateDimensions();
}

// Precondition: 0 <= rectangleId < numRectangles() and
//     heightFactor > 0 and widthFactor > 0
// Scales the Rectangle's dimensions by the specified amounts.
// Postcondition: array[rectangleId]'s new height = old height * heightFactor
//    and array[rectangleId]'s new width = old width * widthFactor and
//    Canvas dimensions may have been altered as a result
void Canvas::scale(int rectangleId, float heightFactor, float widthFactor) {
  if (rectangleId < 0 || rectangleId >= n_Rect) {
    return;
  }

  Rectangle &rectangle = rectangles[rectangleId];
  rectangle.scale(heightFactor, widthFactor);

  updateDimensions();
}

// Precondition: 0 <= rectangleId < numRectangles()
// Postcondition: array[rectangleId] now has the new colour.
void Canvas::change(int rectangleId, Colour newColour) {
  if (rectangleId < 0 || rectangleId >= n_Rect) {
    return;
  }

  Rectangle &rectangle = rectangles[rectangleId];
  rectangle.change(newColour);
}

// Precondition: 0 <= rectangleId < numRectangles()
// Postcondition: For all rectangles with rectangleId < idx < numRectangles,
//    idx = idx-1 and numRectangles() = numRectangles()-1
void Canvas::remove(int rectangleId) {
  if (rectangleId < 0 || rectangleId >= n_Rect) {
    return;
  }

  n_Rect--;
  for (int i = rectangleId; i < n_Rect; i++) {
    rectangles[i] = rectangles[i + 1];
  }
  updateDimensions();
}

void Canvas::updateDimensions() {
  int newHeight = 0, newWidth = 0;
  for (int i = 0; i < n_Rect; i++) {
    newWidth = std::max(newWidth, rectangles[i].getPoint().getX() +
                                      rectangles[i].getWidth());
    newHeight = std::max(newHeight, rectangles[i].getPoint().getY() +
                                        rectangles[i].getHeight());
  }

  height = newHeight;
  width = newWidth;
}

// Removes all rectangles from the canvas, setting height = width = 0.
void Canvas::empty() {
  height = 0;
  width = 0;
  n_Rect = 0;
  c_Rect = DEFAULT_RECT_SIZE;
  delete[] rectangles;
  rectangles = new Rectangle[DEFAULT_RECT_SIZE];
}

std::ostream &operator<<(std::ostream &out, const Canvas &canvas) {
  out << "Dimensions: " << canvas.getHeight() << "x" << canvas.getWidth()
      << "\n";
  out << "Number of rectangles: " << canvas.numRectangles() << "\n";
  for (int i = 0; i < canvas.numRectangles(); i++) {
    out << '\t' << canvas.get(i) << "\n";
  }
  return out;
}
