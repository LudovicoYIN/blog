---
title: C++面向对象
date: 2023-04-13
sidebar: auto
categories:
  - c++
tags:
  - OOP
---

## 面向对象（Object-oriented programming，OOP）
### 概念
 面向对象（Object-oriented programming，OOP）的概念，是一种抽象概念，意思是在程序设计中可以将你的问题抽象成一个对象。每个对象都是一个独立的实体，它具有自己的属性与方法，可以和其他对象进行交互与通信。
### 以游戏人物为例子
 我们可以将每个游戏人物都看作一个对象，每个游戏人物都具有自己的属性和方法。
 以下是一个对象中可能具有的属性和方法：

属性：
- 名字
- 生命值
- 能量值
- 等级
- 经验值
  
方法：
- 攻击
- 防御
- 升级
  
### Code（OOP）
``` cpp 
 class GameCharacter {
  private:
    // 属性
    std::string name;
    int health;
    int energy;
    int level;
    int experience;
    
  public:
    // 构造函数
    GameCharacter(std::string n, int h, int e, int l, int exp) :
        name(n), health(h), energy(e), level(l), experience(exp) {}
        
    // getter and setter 方法
    std::string getName() const { return name; }
    void setName(std::string n) { name = n; }
    int getHealth() const { return health; }
    void setHealth(int h) { health = h; }
    int getEnergy() const { return energy; }
    void setEnergy(int e) { energy = e; }
    int getLevel() const { return level; }
    void setLevel(int l) { level = l; }
    int getExperience() const { return experience; }
    void setExperience(int exp) { experience = exp; }

    void attack(GameCharacter& enemy);
    void levelUp();
};
```
## 访问权限
在下一步之前需要提一下，C++中，对象的访问权限有三种，分别是public、private和protected。

### 概念
- public表示公开的，意味着可以在任何地方访问，包括类内部和类外部。
- private表示私有的，意味着只能在类的内部访问，类外部无法访问。
- protected表示受保护的，意味着只能在类的内部以及其派生类中访问。

### 访问权限表格
根据访问权限总结出不同的访问类型，如下所示：
|  访问  | public  | protected  | private  |
|  ----  | ----  |  ----  | ----  |
| 同一个类  | yes | yes  | yes |
| 派生类  | yes | yes  | no |
| 外部的类  | yes | no  | no |

默认情况下，C++ 中的成员变量和成员函数是私有的。因此，如果要让类中的成员变量或成员函数在类外部可以访问，必须将其声明为 public。
使用不同的访问权限可以帮助我们控制数据的访问范围，从而提高代码的安全性和可维护性。例如，私有成员可以防止类外部的代码直接修改类中的数据，保护数据的完整性。另外，使用继承时，可以通过不同的访问权限来控制子类是否能够访问父类中的成员。
## OOP中的继承（Inheritance）
### 概念
在OOP中，继承是一种将已有类的属性和方法引入到新的类中的机制。它是面向对象编程中的重要概念之一，也是实现代码复用和提高代码可维护性的关键手段之一。
### 基类 & 派生类
通过继承，一个类可以派生出一个新的类，新类会自动拥有原类的属性和方法。在C++中，继承 = 派生类名+基类的名称和访问权限。基类是被继承的类，而派生类是继承基类而得到的新类。形式如下:
~~~ cpp
class derived-class: access-specifier base-class
~~~
其中，访问修饰符 access-specifier包括（public、protected、private ），base-class 是之前定义过的某个类的名称。如果未使用访问修饰符 access-specifier，则默认为 private。以游戏人物为例子:
### Code（Inheritance）
~~~ cpp
#include <iostream>
#include <string>
using namespace std;

// 基类，游戏人物类
class GameCharacter {
public:
    GameCharacter(string n, int hp, int atk) : name(n), healthPoints(hp), attack(atk) {}
    virtual void attackEnemy() = 0; // 纯虚函数
protected:
    string name; // 游戏人物名称
    int healthPoints; // 生命值
    int attack; // 攻击力
};

// 派生类，战士类
class Warrior : public GameCharacter {
public:
    Warrior(string n, int hp, int atk, int r) : GameCharacter(n, hp, atk), rage(r) {}
    void attackEnemy() override {
        cout << name << "使用剑攻击敌人，造成" << attack << "点伤害" << endl;
    }
    void useSkill() {
        if (rage >= 100) {
            cout << name << "使用愤怒打击，造成" << attack * 2 << "点伤害" << endl;
            rage -= 100;
        } else {
            cout << name << "愤怒值不足，无法使用技能" << endl;
        }
    }
private:
    int rage; // 愤怒值
};

int main() {
    GameCharacter* gc1 = new Warrior("Arthur", 100, 20, 0);
    gc1.attackEnemy();
    Warrior* w1 = dynamic_cast<Warrior*>(gc1);
    w1.useSkill();
    delete gc1;
    return 0;
}
~~~
以上代码运行结果：
~~~
Arthur使用剑攻击敌人，造成20点伤害
Arthur愤怒值不足，无法使用技能
~~~
### 继承类型
C++中的继承方式包括公有继承、私有继承和保护继承。不同的继承方式会影响到派生类对基类成员的访问权限。具体而言：
- 公有继承（public inheritance）：派生类可以访问基类中的公有成员和保护成员，但不能访问基类中的私有成员。
- 私有继承（private inheritance）：派生类可以访问基类中的公有成员、保护成员和私有成员，但这些成员在派生类中都变成了私有成员。
- 保护继承（protected inheritance）：派生类可以访问基类中的公有成员和保护成员，但这些成员在派生类中都变成了保护成员，不能被外部访问。

我们几乎不使用 protected 或 private 继承，通常使用 public 继承。通过继承，派生类可以获得基类的所有属性和方法，同时也可以通过重写基类的虚函数来实现多态性。此外，派生类还可以添加自己的新属性和方法，从而实现对基类的扩展和改进。继承是OOP中非常重要的概念，它可以有效地提高代码复用性和可维护性，是面向对象编程的核心之一。

### 多继承（Multiple Inheritance）
允许一个派生类继承自多个基类。以游戏角色为例，我们可以定义多个基类来描述不同方面的角色属性，然后让一个派生类继承这些基类，从而组合出一个完整的游戏角色。
### Code（Multiple Inheritance）
~~~ cpp
#include <iostream>
using namespace std;

// 角色的基类
class Character {
public:
    virtual void printDescription() {
        cout << "I am a character." << endl;
    }
};

// 攻击属性的基类
class Attack {
public:
    virtual void attack() {
        cout << "I am attacking." << endl;
    }
};

// 魔法属性的基类
class Magic {
public:
    virtual void castSpell() {
        cout << "I am casting a spell." << endl;
    }
};

// 战士角色类，同时继承自Character和Attack两个基类
class Warrior : public Character, public Attack {
public:
    void printDescription() {
        cout << "I am a warrior." << endl;
    }
};

// 法师角色类，同时继承自Character和Magic两个基类
class Wizard : public Character, public Magic {
public:
    void printDescription() {
        cout << "I am a wizard." << endl;
    }
};

// 剑法大师角色类，同时继承自Warrior和Magic两个角色类
class SwordMaster : public Warrior, public Magic {
public:
    void printDescription() {
        cout << "I am a sword master." << endl;
    }
};

int main() {
    Warrior warrior;
    warrior.printDescription();
    warrior.attack();

    Wizard wizard;
    wizard.printDescription();
    wizard.castSpell();

    SwordMaster swordMaster;
    swordMaster.printDescription();
    swordMaster.attack();
    swordMaster.castSpell();

    return 0;
}
~~~

以上代码运行结果：
~~~
I am a warrior.
I am attacking.
I am a wizard.
I am casting a spell.
I am a sword master.
I am attacking.
I am casting a spell.
~~~
## OOP中的多态（Polymorphism）
### 概念
C++的多态性（polymorphism）允许我们使用一个基类指针来调用派生类对象的方法。这个过程中，由于基类指针可以指向不同的派生类对象，因此可以实现不同对象之间的动态绑定。以游戏人物为例子:
### Code（Polymorphism）
~~~ cpp
#include <iostream>
using namespace std;

// 角色的基类
class Character {
public:
    virtual void printDescription() {
        cout << "I am a character." << endl;
    }
};

// 攻击属性的基类
class Attack {
public:
    virtual void attack() {
        cout << "I am attacking." << endl;
    }
};

// 魔法属性的基类
class Magic {
public:
    virtual void castSpell() {
        cout << "I am casting a spell." << endl;
    }
};

// 战士角色类，同时继承自Character和Attack两个基类
class Warrior : public Character, public Attack {
public:
    void printDescription() {
        cout << "I am a warrior." << endl;
    }
};

// 法师角色类，同时继承自Character和Magic两个基类
class Wizard : public Character, public Magic {
public:
    void printDescription() {
        cout << "I am a wizard." << endl;
    }
};

// 剑法大师角色类，同时继承自Warrior和Magic两个角色类
class SwordMaster : public Warrior, public Magic {
public:
    void printDescription() {
        cout << "I am a sword master." << endl;
    }
};

int main() {
    Warrior warrior;
    warrior.printDescription();
    warrior.attack();

    Wizard wizard;
    wizard.printDescription();
    wizard.castSpell();

    SwordMaster swordMaster;
    swordMaster.printDescription();
    swordMaster.attack();
    swordMaster.castSpell();

    return 0;
}
~~~
以上代码运行结果：
~~~
I am a warrior.
I am a wizard.
~~~
## OOP中的数据封装
### 概念
数据封装是一种将数据和操作数据的函数封装在一起的机制，它可以将类的数据成员和成员函数限制在类的内部，只提供公共接口来访问这些成员，从而保护数据不被外部直接访问和修改。数据封装也是面向对象编程的重要概念之一，它可以有效地提高代码的可维护性、可扩展性和安全性。
以游戏人物为例子:
### Code（Polymorphism）
~~~ cpp
#include <iostream>
#include <string>

using namespace std;

class GameCharacter {
private:
    string name;    // 角色名字
    int health;     // 角色生命值
    int damage;     // 角色攻击力

public:
    GameCharacter(string name, int health, int damage) {
        this->name = name;
        this->health = health;
        this->damage = damage;
    }

    // 攻击另一个角色
    void attack(GameCharacter& other) {
        cout << name << " attacks " << other.name << " for " << damage << " damage!" << endl;
        other.health -= damage;
        if (other.health <= 0) {
            cout << other.name << " has been defeated!" << endl;
        }
    }

    // 获取角色生命值
    int getHealth() const {
        return health;
    }

    // 设置角色生命值
    void setHealth(int health) {
        this->health = health;
    }

    // 获取角色攻击力
    int getDamage() const {
        return damage;
    }

    // 设置角色攻击力
    void setDamage(int damage) {
        this->damage = damage;
    }
};

int main() {
    GameCharacter player("Player", 100, 10);
    GameCharacter enemy("Enemy", 50, 5);

    player.attack(enemy);
    enemy.attack(player);

    return 0;
}
~~~
在这个例子中，我们定义了一个游戏角色类GameCharacter，它有三个私有数据成员name、health和damage，以及三个公共成员函数attack、getHealth和getDamage和一个setHealth，attack用于攻击其他角色，getHealth和getDamage用于获取角色的健康值和伤害值，setHealth用于设置角色的健康值。

由于name、health和damage是私有数据成员，外部程序无法直接访问它们。而attack、getHealth、getDamage和setHealth是公共成员函数，可以被外部程序访问。这样，我们就实现了对游戏角色类的数据封装，只暴露了它的公共接口，而隐藏了内部数据成员。以上代码运行结果：
~~~
Player attacks Enemy for 10 damage!
Enemy attacks Player for 5 damage!
~~~

## OOP中的数据抽象
### 概念
数据抽象通过将类的实现细节隐藏起来，只暴露出类的公共接口，从而实现了对类的抽象。

例如生活中的例子：Chatgpt，你可以使用它，但是无需知道它的内部实现原理。

数据抽象主要通过封装实现。封装是一种将数据和方法组合在一起的机制，它可以将数据成员和成员函数封装在类的内部，只提供公共接口来访问这些成员。这样，外部程序就无法直接访问类的内部数据成员，从而实现了数据抽象。
### Code
~~~ cpp
#include <iostream>
#include <string>

using namespace std;

class GameCharacter {
public:
    // 构造函数
    GameCharacter(string name, int health, int damage)
        : name(name), health(health), damage(damage) {}

    // 攻击另一个角色
    void attack(GameCharacter& other) {
        cout << name << " attacks " << other.getName() << " for " << damage << " damage!" << endl;
        other.takeDamage(damage);
        if (other.isDead()) {
            cout << other.getName() << " has been defeated!" << endl;
        }
    }

    // 获取角色名字
    string getName() const {
        return name;
    }

    // 获取角色生命值
    int getHealth() const {
        return health;
    }

    // 获取角色攻击力
    int getDamage() const {
        return damage;
    }

private:
    string name;    // 角色名字
    int health;     // 角色生命值
    int damage;     // 角色攻击力

    // 受到伤害
    void takeDamage(int damage) {
        health -= damage;
    }

    // 是否死亡
    bool isDead() const {
        return health <= 0;
    }
};

int main() {
    GameCharacter player("Player", 100, 10);
    GameCharacter enemy("Enemy", 50, 5);

    player.attack(enemy);
    enemy.attack(player);

    return 0;
}

~~~
在这个例子中，数据抽象体现在类的设计上。我们将角色的生命值、攻击力和是否死亡等数据封装在类的私有部分，隐藏了这些数据的具体实现细节，只提供了攻击和获取角色信息的公共接口，即在抽象层面上表示角色的数据。这样，其他类或函数只能通过这些公共接口来访问和操作角色的数据，而不能直接访问和修改私有数据，从而实现了数据抽象。以上代码运行结果：
~~~
Player attacks Enemy for 10 damage!
Enemy attacks Player for 5 damage!
~~~
## OOP中的抽象（Abstraction）
### 概念
接口描述了类的行为和功能，而不需要完成类的特定实现。

C++ 接口是使用抽象类来实现的，抽象类与数据抽象互不混淆，数据抽象是一个把实现细节与相关的数据分离开的概念。

如果类中至少有一个函数被声明为纯虚函数，则这个类就是抽象类。纯虚函数是通过在声明中使用 "= 0" 来指定的，如下所示：
~~~ cpp
class Box{
   public:
      // 纯虚函数
      virtual double getVolume() = 0;
   private:
      double length;      // 长度
      double breadth;     // 宽度
      double height;      // 高度};
~~~
设计抽象类（通常称为 ABC）的目的，是为了给其他类提供一个可以继承的适当的基类。抽象类不能被用于实例化对象，它只能作为接口使用。如果试图实例化一个抽象类的对象，会导致编译错误。

因此，如果一个 ABC 的子类需要被实例化，则必须实现每个虚函数，这也意味着 C++ 支持使用 ABC 声明接口。如果没有在派生类中重载纯虚函数，就尝试实例化该类的对象，会导致编译错误。
### Code（Abstraction）
~~~ cpp
#include <iostream>
#include <string>

using namespace std;

// 游戏角色接口（抽象类）
class GameCharacter {
public:
    // 析构函数（虚函数）
    virtual ~GameCharacter() {}

    // 攻击另一个角色（纯虚函数）
    virtual void attack(GameCharacter& other) = 0;

    // 获取角色名字（虚函数）
    virtual string getName() const {
        return name;
    }

    // 获取角色生命值（纯虚函数）
    virtual int getHealth() const = 0;

    // 获取角色攻击力（纯虚函数）
    virtual int getDamage() const = 0;

protected:
    string name;    // 角色名字
};

// 玩家角色
class Player : public GameCharacter {
public:
    // 构造函数
    Player(string name, int health, int damage)
        : GameCharacter(), health(health), damage(damage) {
        this->name = name;
    }

    // 攻击另一个角色
    void attack(GameCharacter& other) override {
        cout << name << " attacks " << other.getName() << " for " << damage << " damage!" << endl;
        other.takeDamage(damage);
        if (other.isDead()) {
            cout << other.getName() << " has been defeated!" << endl;
        }
    }

    // 获取角色生命值
    int getHealth() const override {
        return health;
    }

    // 获取角色攻击力
    int getDamage() const override {
        return damage;
    }

private:
    int health;     // 角色生命值
    int damage;     // 角色攻击力

    // 受到伤害
    void takeDamage(int damage) {
        health -= damage;
    }

    // 是否死亡
    bool isDead() const {
        return health <= 0;
    }
};
~~~

~~~

~~~