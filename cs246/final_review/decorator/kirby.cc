#include <iostream>
#include <memory>

// Base character class
class Character {
public:
    virtual ~Character() = default;
    virtual void action() const = 0;
};

// Concrete Character class representing Kirby
class Kirby : public Character {
public:
    void action() const override {
        std::cout << "Kirby performs a basic action." << std::endl;
    }
};

// Base decorator class
class Decorator : public Character {
protected:
    std::shared_ptr<Character> character;

public:
    Decorator(std::shared_ptr<Character> c) : character(c) {}

    virtual void action() const override {
        character->action();
    }
};


// Fill in concrete decorator for Fire ability
class FireDecorator : public Decorator {
public:
    FireDecorator(std::shared_ptr<Character> c) : Decorator(c) {}
    void action() const override {
        Decorator::action();
        std::cout << "Kirby uses a Fire Ability!\n";
    }
};

// Fill in oncrete decorator for Ice ability
class IceDecorator : public Decorator {
public:
    IceDecorator(std::shared_ptr<Character> c) : Decorator(c) {}
    void action() const override {
        Decorator::action();
        std::cout << "Kirby uses a Ice Ability!\n";
    }
};

int main() {
    std::shared_ptr<Character> kirby = std::make_shared<Kirby>();
    kirby->action();
    std::shared_ptr<Character> fireKirby = std::make_shared<FireDecorator>(kirby);
    fireKirby->action();
    std::shared_ptr<Character> iceKirby = std::make_shared<IceDecorator>(kirby);
    iceKirby->action();
    std::shared_ptr<Character> bothKirby = std::make_shared<IceDecorator>(kirby);
    bothKirby = std::make_shared<FireDecorator>(bothKirby);
    bothKirby->action();
}
