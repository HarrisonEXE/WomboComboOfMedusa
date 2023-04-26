/**
 * Modified by Raghavasimhan Sankaranarayanan on 04/08/21
 *
 * Real Time Protocol Music Instrument Digital Interface Daemon
 * Copyright (C) 2019 David Moreno Montero <dmoreno@coralbits.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <csignal>

#include "./config.hpp"
#include "./control_socket.hpp"
#include "./rtpmidid.hpp"
#include <rtpmidid/logger.hpp>
#include <rtpmidid/poller.hpp>

static bool exiting = false;

void sigterm_f(int) {
    if (exiting) {
        exit(1);
    }
    exiting = true;
    INFO("SIGTERM received. Closing.");
    rtpmidid::poller.close();
}

void sigint_f(int) {
    if (exiting) {
        exit(1);
    }
    exiting = true;
    INFO("SIGINT received. Closing.");
    rtpmidid::poller.close();
}

int main(int argc, char **argv) {

    // We dont need crypto rand, just some rand
    srand(time(nullptr));

    signal(SIGINT, sigint_f);
    signal(SIGTERM, sigterm_f);

    INFO("Real Time Protocol Music Instrument Digital Interface Daemon - {}",
         rtpmidid::VERSION);
    INFO("(C) 2019 David Moreno Montero <dmoreno@coralbits.com>");

    auto options = rtpmidid::parse_cmd_args(argc - 1, argv + 1);

    try {
        auto rtpmidid = rtpmidid::rtpmidid_t(&options);
        auto control = rtpmidid::control_socket_t(rtpmidid, options.control);

        while (rtpmidid::poller.is_open()) {
            rtpmidid::poller.wait();
        }
    } catch (const std::exception &e) {
        ERROR("{}", e.what());
        return 1;
    }
    return 0;
}
