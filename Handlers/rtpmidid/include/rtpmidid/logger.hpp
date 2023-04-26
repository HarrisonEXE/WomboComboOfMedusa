/**
 * Real Time Protocol Music Instrument Digital Interface Daemon
 * Copyright (C) 2019-2020 David Moreno Montero <dmoreno@coralbits.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#pragma once
#include <fmt/format.h>

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED true
#endif

#ifndef INFO_ENABLED
#define INFO_ENABLED true
#endif

#if DEBUG_ENABLED
#define DEBUG(...) logger::log(__FILE__, __LINE__, logger::DEBUG, __VA_ARGS__)
#else
#define DEBUG(...) false
#endif
#define WARNING(...)                                                           \
  logger::log(__FILE__, __LINE__, logger::WARNING, __VA_ARGS__)
#define ERROR(...) logger::log(__FILE__, __LINE__, logger::ERROR, __VA_ARGS__)
#if INFO_ENABLED
#define INFO(...) logger::log(__FILE__, __LINE__, logger::INFO, __VA_ARGS__)
#else
#define INFO(...) false
#endif
#define SUCCESS(...)                                                           \
  logger::log(__FILE__, __LINE__, logger::SUCCESS, __VA_ARGS__)

namespace logger {
class logger;

extern logger __logger;

enum LogLevel {
  DEBUG,
  WARNING,
  ERROR,
  INFO,
  SUCCESS,
};

class logger {
public:
  logger();
  ~logger();

  void log(const char *filename, int lineno, LogLevel loglevel,
           const std::string &msg);
  void flush();
};

template <typename... Args>
inline void log(const char *fullpath, int lineno, LogLevel loglevel,
                Args... args) {

  // Get ony the file name part, not full path. Assumes a / and ends in 0.
  const char *filename = fullpath;
  while (*filename)
    ++filename;
  while (*filename != '/')
    --filename;
  ++filename;

  auto str = fmt::format(args...);
  __logger.log(filename, lineno, loglevel, str);
}

inline void flush() { __logger.flush(); }
} // namespace logger
