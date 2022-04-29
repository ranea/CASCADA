header= r"""
struct preimage_8bit { unsigned char i; unsigned char j; _Bool found; };
struct preimage_8bit find_RX_preimage_8bit(unsigned char, unsigned char, unsigned char);
unsigned int count_RX_preimage_8bit(unsigned char, unsigned char, unsigned char);
struct preimage_16bit { unsigned short i; unsigned short j; _Bool found; };
struct preimage_16bit find_RX_preimage_16bit(unsigned short, unsigned short, unsigned short);
unsigned int count_RX_preimage_16bit(unsigned short, unsigned short, unsigned short);
"""

source = r"""
// #include <stdint.h>
#define SIZE 256
#define SIZEv16 65536
// #define ROTL1(a) (a << 1) | (a >> 7)
// #define ROTR1(a) (a >> 1) | (a << 7)

struct preimage_8bit { unsigned char i; unsigned char j;_Bool found; };

static inline unsigned char rotl1(unsigned char x){
    return (x << 1) | (x >> 7);
}

static inline unsigned char rotr1(unsigned char x){
    return (x >> 1) | (x << 7);
}

static inline unsigned char eval(unsigned char x0, unsigned char x1, unsigned char id0, unsigned char id1){
    unsigned char fx = (x0 + x1) % SIZE;  
    unsigned char y0 = rotl1(x0) ^ id0;
    unsigned char y1 = rotl1(x1) ^ id1;
    unsigned char fy  = (y0 + y1) % SIZE;
    return rotl1(fx) ^ fy;
}

static struct preimage_8bit find_RX_preimage_8bit(unsigned char a0, unsigned char a1, unsigned char b)
{
    struct preimage_8bit p = {.i=0, .j=0, .found=0};
    unsigned char i = 0;
    unsigned char j = 0;

    do
    {
        j = 0;
        do
        {
            if( eval(i, j, a0, a1) == b){
                p.i = i;
                p.j = j;
                p.found = 1;
                return p;
            }
        } while (++j != 0);
    } while (++i != 0);

    return p;
}

static unsigned int count_RX_preimage_8bit(unsigned char a0, unsigned char a1, unsigned char b)
{
    unsigned int num_preimages = 0;
    unsigned char i = 0;
    unsigned char j = 0;

    do
    {
        j = 0;
        do
        {
            if( eval(i, j, a0, a1) == b){
                num_preimages += 1;
            }
        } while (++j != 0);
    } while (++i != 0);

    return num_preimages;
}

struct preimage_16bit { unsigned short i; unsigned short j;_Bool found; };

static inline unsigned short rotl1v16(unsigned short x){
    return (x << 1) | (x >> 15);
}

static inline unsigned short rotr1v16(unsigned short x){
    return (x >> 1) | (x << 15);
}

static inline unsigned short evalv16(unsigned short x0, unsigned short x1, unsigned short id0, unsigned short id1){
    unsigned short fx = (x0 + x1) % SIZEv16;  
    unsigned short y0 = rotl1v16(x0) ^ id0;
    unsigned short y1 = rotl1v16(x1) ^ id1;
    unsigned short fy  = (y0 + y1) % SIZEv16;
    return rotl1v16(fx) ^ fy;
}

static struct preimage_16bit find_RX_preimage_16bit(unsigned short a0, unsigned short a1, unsigned short b)
{
    struct preimage_16bit p = {.i=0, .j=0, .found=0};
    unsigned short i = 0;
    unsigned short j = 0;

    do
    {
        j = 0;
        do
        {
            if( evalv16(i, j, a0, a1) == b){
                p.i = i;
                p.j = j;
                p.found = 1;
                return p;
            }
        } while (++j != 0);
    } while (++i != 0);

    return p;
}

static unsigned int count_RX_preimage_16bit(unsigned short a0, unsigned short a1, unsigned short b)
{
    unsigned int num_preimages = 0;
    unsigned short i = 0;
    unsigned short j = 0;

    do
    {
        j = 0;
        do
        {
            if( evalv16(i, j, a0, a1) == b){
                num_preimages += 1;
            }
        } while (++j != 0);
    } while (++i != 0);

    return num_preimages;
}
"""
