#pragma once

#include "profile.h"

bool enabled = false;
bool started = false;

void profile_start() { enabled = true; }

void profile_stop() {
    enabled = false;
    started = false;
}
