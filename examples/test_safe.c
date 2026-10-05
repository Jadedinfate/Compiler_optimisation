#include <stdio.h>

// A perfectly safe function that just adds two fixed numbers
int calculate_sum(int a, int b) {
    int total = a + b;
    return total;
}

int main() {
    int x = 5;
    int y = 10;
    int sum = calculate_sum(x, y);
    
    printf("The sum is: %d\n", sum);
    
    return 0;
}