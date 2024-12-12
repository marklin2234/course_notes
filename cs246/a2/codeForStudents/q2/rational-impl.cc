module rational;

import <iostream>;

Rational::Rational(int num, int den) {
  this->num = num;
  this->den = den;
}

Rational Rational::operator+(const Rational &rhs) const {
  Rational ret;
  ret.num = (this->num * rhs.den) + (this->den * rhs.num);
  ret.den = this->den * rhs.den;
  ret.simplify();
  return ret;
}
Rational Rational::operator-(const Rational &rhs) const {
  Rational ret;
  ret.num = (this->num * rhs.den) - (this->den * rhs.num);
  ret.den = this->den * rhs.den;
  ret.simplify();
  return ret;
}
Rational Rational::operator*(const Rational &rhs) const {
  Rational ret;
  ret.num = this->num * rhs.num;
  ret.den = this->den * rhs.den;
  ret.simplify();
  return ret;
}
Rational Rational::operator/(const Rational &rhs) const {
  Rational ret;
  ret.num = this->num * rhs.den;
  ret.den = this->den * rhs.num;
  ret.simplify();
  return ret;
}
Rational Rational::operator-() const {
  Rational ret = *this;
  ret.num *= -1;
  return ret;
}

Rational &Rational::operator+=(const Rational &rhs) {
  *this = *this + rhs;
  this->simplify();
  return *this;
}
Rational &Rational::operator-=(const Rational &rhs) {
  *this = *this - rhs;
  this->simplify();
  return *this;
}

int gcd(int a, int b) {
  if (a == 0) {
    return b;
  } else if (b == 0) {
    return a;
  } else if (a == b) {
    return a;
  }

  if (a > b) {
    return gcd(a - b, b);
  }
  return gcd(a, b - a);
}

void Rational::simplify() {
  int n;
  bool isNeg = this->num < 0;
  if (this->num < 0) {
    this->num *= 1;
  }
  while ((n = gcd(this->num, this->den)) != 1) {
    this->num /= n;
    this->den /= n;
  }
  if (isNeg) {
    this->num *= -1;
  }
}

int Rational::getNumerator() const { return this->num; }
int Rational::getDenominator() const { return this->den; }
bool Rational::isZero() const { return this->num == 0; }

std::ostream &operator<<(std::ostream &out, const Rational &rat) {
  if (rat.den == 1 || rat.num == 0) {
    out << rat.num << "\n";
  } else {
    out << rat.num << "/" << rat.den;
  }
  return out;
}

std::istream &operator>>(std::istream &in, Rational &rat) {
  int num, den;
  char slash;

  in >> num >> slash >> den;
  if (den < 0) {
    num *= -1;
    den *= -1;
  }
  rat.num = num;
  rat.den = den;
  return in;
}
