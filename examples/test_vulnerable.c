// Sample vulnerable C code containing classic buffer overflow vulnerability
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void vulnerable_function(char *user_input) {
    char buffer[64];
    // Unsafe string copy without boundary check causes stack buffer overflow
    strcpy(buffer, user_input);
    printf("Copied buffer: %s\n", buffer);
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    }
    return 0;
}
