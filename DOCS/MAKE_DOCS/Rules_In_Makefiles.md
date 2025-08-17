## What are Rules?

A **rule** in a Makefile tells `make` the following:

1. **What to build** → the **target**
2. **What it depends on** → the **prerequisites (dependencies)**
3. **How to build it** → the **recipe (commands)**

### General Syntax

```make
target: prerequisites
    recipe
```

## Example Rule

```make
main.o: main.c main.h
    gcc -c main.c -o main.o
```

In the above example Rule the Target, Prerequisites and Recipe is as follows:

1. **Target**: main.o (object file)
2. **Prerequisites**: main.c, main.h
3. **Recipe**: Compile main.c into main.o

!> If either main.c or main.h changes, main.o will be rebuilt.
