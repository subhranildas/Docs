## What is Make?

**Make** is a powerful build automation tool that simplifies the process of compiling and linking programs.  
It was originally created for **C projects**, but it can be used for any task where one needs to **define dependencies and automate commands**.

At its core, `make` takes care of the following things:

- It decides **when** a file needs to be rebuilt (based on timestamps).
- It executes the correct **commands** to update the file.
- It automates repetitive tasks like compiling, linking, testing, packaging, or cleaning.

---

## What is a Makefile?

A **Makefile** is a plain-text file that contains a set of rules and instructions for `make` to follow.  
A Makefile typically defines the following:

- **Targets** (files or actions to create)
- **Dependencies** (files needed before building a target)
- **Recipes** (shell commands to build the target)

?> In later secions it is discussed in details.

### Example

```make
program: main.o utils.o
    gcc main.o utils.o -o program

main.o: main.c main.h
    gcc -c main.c -o main.o

utils.o: utils.c utils.h
    gcc -c utils.c -o utils.o

.PHONY: clean
clean:
    rm -f *.o program
```

### Explanation

1. **program: main.o utils.o**  
   Builds the `program` executable from `main.o` and `utils.o` using `gcc`.

2. **main.o: main.c main.h**  
   Compiles `main.c` into `main.o` if either `main.c` or `main.h` has changed.

3. **utils.o: utils.c utils.h**  
   Compiles `utils.c` into `utils.o` if either `utils.c` or `utils.h` has changed.

4. **.PHONY: clean**  
   Declares `clean` as a special target, so it always runs when requested.

5. **clean:**  
   Removes all object files (`*.o`) and the `program` executable to clean the build
