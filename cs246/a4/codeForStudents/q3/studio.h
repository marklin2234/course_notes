#ifndef STUDIO_H
#define STUDIO_H
#include "subject.h"
#include <iostream>

class AsciiArt;

class Studio : public Subject {
  AsciiArt *thePicture;
  int ticks = 0;

public:
  explicit Studio(AsciiArt *picture) : thePicture{picture} {}

  AsciiArt *&picture() { return thePicture; }
  void reset();
  void render();
  void animate(int numTicks);

  char getState(int row, int col) const override;

  ~Studio();
};

#endif
