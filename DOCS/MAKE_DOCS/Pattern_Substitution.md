## Substitutions in Makefiles

Makefiles provide **substitution mechanisms** to transform lists of text (often filenames) into new forms.  
This is especially useful when dealing with object files, source files, or generated targets.

## The Basics of Substitution

The most common form of substitution in Makefiles uses the **pattern substitution operator**:

```makefile
$(VAR:SUFFIX=NEW_SUFFIX)
```

This means:

- Take each word in `$(VAR)`
- Replace `SUFFIX` with `NEW_SUFFIX`
- Produce the resulting list

### Example: Source → Object Files

```makefile
SRC = main.c utils.c file.c
OBJ = $(SRC:.c=.o)

all:
	@echo "Sources: $(SRC)"
	@echo "Objects: $(OBJ)"
```

Output:

```bash
Sources: main.c utils.c file.c
Objects: main.o utils.o file.o
```

Here, .c is replaced by .o for every file in SRC.

## Pattern Substitution

More generally, substitutions can use a % wildcard for flexible transformations:

```makefile
$(VAR:PATTERN=REPLACEMENT)
```

PATTERN may include %, which matches part of each word.
REPLACEMENT uses % to substitute the matched part.

### Example: Rename Files with %

```makefile
SRC = foo.c bar.c baz.c
OBJ = $(SRC:%.c=build/%.o)

all:
@echo $(OBJ)
```

Output:

```bash
build/foo.o build/bar.o build/baz.o
```

## Using patsubst Function

GNU Make also provides the explicit function:

```makefile
$(patsubst PATTERN,REPLACEMENT,TEXT)
```

This is equivalent to the :pattern=replacement shorthand, but more versatile.

### Example: Using patsubst

```makefile
SRC = one.cpp two.cpp three.cpp
OBJ = $(patsubst %.cpp,%.o,$(SRC))

all:
	@echo $(OBJ)

```

Output:

```bash
one.o two.o three.o
```

## Multiple Substitutions

Substitutions can also be chained if needed.

### Example: Chained Substitutions

```makefile
FILES = a.c b.cpp c.c
OBJS  = $(FILES:.c=.o)
OBJS  := $(OBJS:.cpp=.o)

all:
	@echo $(OBJS)
```

Output:

```bash
a.o b.o c.o
```

## Advanced Examples

### Example: Converting Headers to Generated Sources

```makefile
HEADERS = inc/a.h inc/b.h inc/c.h
GENERATED = $(HEADERS:inc/%.h=gen/%.c)

all:
	@echo $(GENERATED)
```

Output:

```bash
gen/a.c gen/b.c gen/c.c
```

### Example: Using basename and addsuffix

Make also provides other text functions that can combine with substitutions.

```makefile
SRC = alpha.c beta.c gamma.c

# Remove suffix and add a new one
NAMES = $(basename $(SRC))
ASM   = $(addsuffix .s,$(NAMES))

all:
	@echo $(ASM)
```

Output:

```bash
alpha.s beta.s gamma.s
```
