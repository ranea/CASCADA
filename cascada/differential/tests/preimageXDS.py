header= r"""
struct preimage { unsigned char i; unsigned char j; _Bool found; };
struct preimage find_XOR_preimage_8bit(unsigned char, unsigned char, unsigned char);
unsigned int count_XOR_preimage_8bit(unsigned char, unsigned char, unsigned char);
"""

source = r"""
// #include <stdint.h>
#define SIZE 256

struct preimage { unsigned char i; unsigned char j;_Bool found; };

static struct preimage find_XOR_preimage_8bit(unsigned char a0, unsigned char a1, unsigned char b)
{
    struct preimage p = {.i=0, .j=0, .found=0};
    unsigned short i = 0;
    unsigned short j = 0;

    do
    {
        j = 0;
        do
        {
            if( ( ((SIZE + i - j) % SIZE ) ^ ( (SIZE + (i ^ a0) - (j ^ a1)) % SIZE) ) == b){
                p.i = i;
                p.j = j;
                p.found = 1;
                return p;
            }
        } while (++j != SIZE);
    } while (++i != SIZE);

    return p;
}

static unsigned int count_XOR_preimage_8bit(unsigned char a0, unsigned char a1, unsigned char b)
{
    unsigned int num_preimages = 0;
    unsigned short i = 0;
    unsigned short j = 0;

    do
    {
        j = 0;
        do
        {
            if( ( ((SIZE + i - j) % SIZE ) ^ ( (SIZE + (i ^ a0) - (j ^ a1)) % SIZE) ) == b){
                num_preimages += 1;
            }
        } while (++j != SIZE);
    } while (++i != SIZE);

    return num_preimages;
}
"""
