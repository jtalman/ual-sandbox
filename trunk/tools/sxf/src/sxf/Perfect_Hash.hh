/* C++ code produced by gperf version 2.7.2 */
/* Command-line: gperf -a -p -j1 -g -o -t -T -k1,2 -L C++ -Z SXF_Perfect_Hash -N sxf_elements_gperf  */
#include "sxf/Def.hh"
/* Command-line: $(GPERF) -a -p  -j1 -g -o -t -T -k1,2 -N sxf_elements_gperf  */ 

#define TOTAL_KEYWORDS 22
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 11
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 25
/* maximum key range = 23, duplicates = 0 */

class SXF_Perfect_Hash
{
private:
  static inline unsigned int hash (const char *str, unsigned int len);
public:
  static struct SXF_Key *sxf_elements_gperf (const char *str, unsigned int len);
};

inline unsigned int
SXF_Perfect_Hash::hash (register const char *str, register unsigned int len)
{
  static unsigned char asso_values[] =
    {
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26,  0, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 19,  0,  4,
      16,  1,  0, 26, 11,  0, 26,  0, 11,  0,
      14,  0, 26,  0,  0, 12, 26,  1,  6, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 26, 26
    };
  return len + asso_values[(unsigned char)str[1]] + asso_values[(unsigned char)str[0]];
}

struct SXF_Key *
SXF_Perfect_Hash::sxf_elements_gperf (register const char *str, register unsigned int len)
{
  static struct SXF_Key wordlist[] =
    {
      {""}, {""}, {""},
      {"$$$", 21},
      {""},
      {"rbend", 2},
      {"kicker", 11},
      {"monitor", 16},
      {"rfcavity", 12},
      {"beambeam", 20},
      {"multipole", 7},
      {"quadrupole", 4},
      {"octupole", 6},
      {"vkicker", 10},
      {"vmonitor", 15},
      {"rcollimator", 19},
      {"ecollimator", 18},
      {"sbend", 3},
      {"hkicker", 9},
      {"hmonitor", 14},
      {"solenoid", 8},
      {"drift", 1},
      {"sextupole", 5},
      {"elseparator", 13},
      {"instrument", 17},
      {"marker", 0,}
    };

  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register int key = hash (str, len);

      if (key <= MAX_HASH_VALUE && key >= 0)
        {
          register const char *s = wordlist[key].name;

          if (*str == *s && !strcmp (str + 1, s + 1))
            return &wordlist[key];
        }
    }
  return 0;
}
