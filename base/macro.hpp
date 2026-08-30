#ifndef _FILO2_MACRO_HPP_
#define _FILO2_MACRO_HPP_

#if defined(_MSC_VER)
    #define likely(condition) (condition)
    #define unlikely(condition) (condition)
    #define __attribute__(x)
#else
    #define likely(condition) __builtin_expect(static_cast<bool>(condition), 1)
    #define unlikely(condition) __builtin_expect(static_cast<bool>(condition), 0)
#endif

#endif