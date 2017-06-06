target = simplexproj.so

all: $(target)

$(target): simplexproj.c
	gcc -O3 -ffast-math -shared -Wl,-soname,$@ -o $@ -fPIC $<

clean:
	rm $(target)
