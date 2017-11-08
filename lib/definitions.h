/******************************************************************************
 * definitions.h
 *
 * Definition of basic types
 ******************************************************************************
 * Copyright (C) 2017 Sebastian Lamm <lamm@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef _DEFINITIONS_H_
#define _DEFINITIONS_H_

// Constants
typedef int PEID;
const PEID ROOT = 0;

// High/low prec
typedef long double HPFloat;
typedef double LPFloat;
typedef long long LONG;
typedef unsigned long long ULONG;

// Graph access
typedef unsigned long long VertexID;
typedef unsigned long long EdgeID;

#endif
