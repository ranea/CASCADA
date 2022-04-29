header= r"""
struct preimage { unsigned char i; unsigned char j; _Bool found; };
struct preimage find_XOR_preimage_8bit(unsigned char, unsigned char, unsigned char);
unsigned int count_XOR_preimage_8bit(unsigned char, unsigned char, unsigned char);
"""

source = r"""
// #include <stdint.h>
#define SIZE 256
// #define ROTL1(a) (a << 1) | (a >> 7)
// #define ROTR1(a) (a >> 1) | (a << 7)

struct preimage { unsigned char i; unsigned char j;_Bool found; };

static inline unsigned char rotl1(unsigned char x){
    return (x << 1) | (x >> 7);  // & 2**8 - 1 (if not casted)
}

static inline unsigned char rotr1(unsigned char x){
    return (x >> 1) | (x << 7);  // & 2**8 - 1 (if not casted)
}

static struct preimage find_XOR_preimage_8bit(unsigned char a0, unsigned char a1, unsigned char b)
{
    struct preimage p = {.i=0, .j=0, .found=0};
    unsigned char i = 0;
    unsigned char j = 0;

    do
    {
        j = 0;
        do
        {
            if( ( ((i + j) % SIZE ) ^ ( ((i ^ a0) + (j ^ a1)) % SIZE) ) == b){
                p.i = i;
                p.j = j;
                p.found = 1;
                return p;
            }
        } while (++j != 0);
    } while (++i != 0);

    return p;
}

static unsigned int count_XOR_preimage_8bit(unsigned char a0, unsigned char a1, unsigned char b)
{
    unsigned int num_preimages = 0;
    unsigned char i = 0;
    unsigned char j = 0;

    do
    {
        j = 0;
        do
        {
            if( ( ((i + j) % SIZE ) ^ ( ((i ^ a0) + (j ^ a1)) % SIZE) ) == b){
                num_preimages += 1;
            }
        } while (++j != 0);
    } while (++i != 0);

    return num_preimages;
}
"""
