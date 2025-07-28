# === Variables de configuración ===
SRC = src/
OBJ = obj/
INC = include/
exe = bin/panorama_extended

# OpenCV detectado automáticamente
CFLAGS = $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)
COMPILER = g++ -O3

# Archivos fuente, objeto y cabeceras
units_cpp := $(wildcard $(addprefix $(SRC), *.cpp))
units_o := $(addprefix $(OBJ), $(addsuffix .o, $(notdir $(basename $(units_cpp)))))
headers := $(wildcard $(addprefix $(INC), *.h*))

# === Reglas principales ===

x: $(exe)
	@echo Compilado: $(exe)

mkdir:
	mkdir -p bin/ obj/

init: mkdir $(exe)
	@echo Compilado: $(exe)

# Enlazar ejecutable con bibliotecas al final
$(exe): $(units_o) makefile
	$(COMPILER) -o $(exe) $(units_o) $(LDFLAGS)

# Compilar cada .cpp en .o
$(OBJ)%.o : $(SRC)%.cpp
	$(COMPILER) -o $@ -c $(CFLAGS) -I$(INC) $<

# Forzar recompilación si cambian headers
$(units_cpp) : $(headers)
	touch $(units_cpp)

# === PROCESAR ARCHIVOS .obj EN PARALELO (7 NÚCLEOS) ===
procesar_obj: $(exe)
	@echo "Procesando archivos .obj en paralelo desde /media/juanan/TOSHIBA EXT/PubisObj/pubis"
	@mkdir -p out/pe out/fm
	@find "/media/juanan/TOSHIBA EXT/PubisObj/pubis" -type f -name "*.obj" | \
	while read -r file; do \
		dir=$$(dirname "$$file"); \
		side=$$(basename "$$dir"); \
		parent=$$(dirname "$$dir"); \
		num=$$(basename "$$parent"); \
		name="$$num"_$$side; \
		echo "$$name|$$file"; \
	done | \
	xargs -P 7 -I{} bash -c ' \
		entry="{}"; \
		name=$${entry%%|*}; \
		file=$${entry#*|}; \
		echo "→ Ejecutando: $(exe) \"$$name\" \"$$file\" out/pe out/fm"; \
		"$(exe)" "$$name" "$$file" "out/pe" "out/fm"; \
	'

# === Documentación con Doxygen ===
doc:
	doxygen Doxyfile

# === Limpiar ===
clean:
	rm -rf obj/*.o bin/* doc/* out/pe/* out/fm/*
