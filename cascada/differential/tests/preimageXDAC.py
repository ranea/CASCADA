header = r"""
struct preimage { unsigned char i; _Bool found; };
struct preimage find_XOR_preimage_8bit(unsigned char, unsigned char, unsigned char);
unsigned int count_XOR_preimage_8bit(unsigned char, unsigned char, unsigned char);
"""

source = r"""
// #include <stdint.h>
#define SIZE 256

struct preimage { unsigned char i; _Bool found; };

static struct preimage find_XOR_preimage_8bit(unsigned char u, unsigned char v, unsigned char a)
{
    struct preimage p = { .i=0, .found=0};
    unsigned char i = 0;
    
    do
    {
        if( ( ((i + a) % SIZE ) ^ ( ((i ^ u) + a) % SIZE) ) == v){
            p.i = i;
            p.found = 1;
            return p;
        }
    } while (++i != 0);
    
    return p;
}

static unsigned int count_XOR_preimage_8bit(unsigned char u, unsigned char v, unsigned char a)
{
  unsigned int num_preimages = 0;
  unsigned char i = 0;

  do
  {
      if( ( ((i + a) % SIZE ) ^ ( ((i ^ u) + a) % SIZE) ) == v){
          num_preimages += 1;
      }
  } while (++i != 0);

  return num_preimages;
}
"""
