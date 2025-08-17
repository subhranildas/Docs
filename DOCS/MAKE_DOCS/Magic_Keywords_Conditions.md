## Magic Keywords and Symbols in Make

Makefiles use special keywords and symbols to automate build processes. Below are common magic keywords and symbols, their meanings, and usage examples.

## Automatic Variables

| Variable | Meaning                         | Example                                                      |
| -------- | ------------------------------- | ------------------------------------------------------------ |
| `$@`     | Target name                     | `echo $@` prints the target's name                           |
| `$<`     | First prerequisite              | `cp $< $@` copies the first prerequisite to the target       |
| `$^`     | All prerequisites               | `cat $^ > $@` concatenates all prerequisites into the target |
| `$?`     | Prerequisites newer than target | `echo $?` lists changed prerequisites                        |
| `$*`     | Stem matched by pattern rule    | Useful in pattern rules: `%.o: %.c`                          |

### Example

```makefile
%.o: %.c
    gcc -c $< -o $@
```

_Compiles `.c` file to `.o` using automatic variables._

## Special Targets

| Target      | Purpose                                  |
| ----------- | ---------------------------------------- |
| `.PHONY`    | Declares phony targets (not files)       |
| `.SUFFIXES` | Defines file suffixes for implicit rules |
| `.DEFAULT`  | Default rule for unknown targets         |
| `.PRECIOUS` | Prevents deletion of intermediate files  |

### Example

```makefile
.PHONY: clean
clean:
    rm -f *.o
```

_Declares `clean` as a phony target._

## Pattern Rules

Use `%` as a wildcard for file names.

```makefile
%.o: %.c
    gcc -c $< -o $@
```

_Builds `.o` from `.c` files automatically._

## Target Directory

Use `$(@D)` to reference the directory part of the target name.

### Example

```makefile
$(TARGET): $(SRCS)
    mkdir -p $(@D)
    gcc $(SRCS) -o $@
```

_Creates the target's directory before building the target._

## Conditional Statements

Conditional statements in Makefiles lets us control which parts of the Makefile are executed based on variable values.
This helps us customize build flags, recipes, or dependencies depending on the build environment or configuration.

### Example 1

```makefile
ifeq ($(DEBUG),yes)
    CFLAGS += -g
endif
```

_Adds debug flags if `DEBUG` is set to `yes`._

### Example 2

```makefile
ifeq ($(TOOLCHAIN),gcc)
    CC = gcc
    BUILD_CMD = $(CC) $(CFLAGS) -o myapp main.c
else ifeq ($(TOOLCHAIN),clang)
    CC = clang
    BUILD_CMD = $(CC) $(CFLAGS) -o myapp main.c
endif

all:
    $(BUILD_CMD)
```

_Uses `gcc` or `clang` as the compiler depending on the `TOOLCHAIN` value._
