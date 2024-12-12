#pragma once

#include <asciiart.h>
class Decorator : public AsciiArt {
protected:
  AsciiArt *art_;

public:
  explicit Decorator(AsciiArt *art);
  virtual ~Decorator();
  virtual char charAt(int row, int col, int tick) override;
};
