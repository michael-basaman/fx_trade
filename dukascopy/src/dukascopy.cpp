/*
Copyright 2013 Michael O'Keeffe (a.k.a. ninety47).

This file is part of ninety47 Dukascopy toolbox.

The "ninety47 Dukascopy toolbox" is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License,
or any later version.

"ninety47 Dukascopy toolbox" is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
"ninety47 Dukascopy toolbox".  If not, see <http://www.gnu.org/licenses/>.
*/

#include "ninety47/dukascopy.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <algorithm>
#include <vector>
#include "ninety47/dukascopy/defs.h"
#include "ninety47/dukascopy/io.hpp"
#include "ninety47/dukascopy/lzma.h"



namespace n47 {

namespace pt = boost::posix_time;


void tickFromBuffer(tick_vector &data, unsigned char *buffer, float digits, size_t offset) {
	bytesTo<unsigned int, n47::BigEndian> bytesTo_unsigned;
	bytesTo<float, n47::BigEndian> bytesTo_float;

	unsigned int ts = bytesTo_unsigned(buffer + offset);
	pt::time_duration ms = pt::millisec(ts);
	unsigned int ofs = offset + sizeof(ts);
	float ask = bytesTo_unsigned(buffer + ofs) * digits;
	ofs += sizeof(ts);
	float bid = bytesTo_unsigned(buffer + ofs) * digits;
	ofs += sizeof(ts);
	float askv = bytesTo_float(buffer + ofs);
	ofs += sizeof(ts);
	float bidv = bytesTo_float(buffer + ofs);

	data.push_back(tick(ms, ask, bid, askv, bidv));
}


void read_bin(tick_vector &data, unsigned char *buffer, size_t buffer_size, float point_value) {
	std::size_t offset = 0;

	while ( offset < buffer_size ) {
		tickFromBuffer(data, buffer, point_value, offset);
		offset += ROW_SIZE;
	}
}


void read_bi5(tick_vector &data, unsigned char *lzma_buffer, size_t lzma_buffer_size, float point_value, size_t *bytes_read) {
	// decompress
	int status;
	unsigned char *buffer = n47::lzma::decompress(lzma_buffer,
			lzma_buffer_size, &status, bytes_read);

	if (status != N47_E_OK) {
		bytes_read = 0;
	} else {
		// convert to tick data (with read_bin).
		read_bin(data, buffer, *bytes_read, point_value);
		delete [] buffer;
	}
}


void read(tick_vector &data, const char *filename, float point_value, size_t *bytes_read) {
	size_t buffer_size = 0;
	unsigned char *buffer = n47::io::loadToBuffer<unsigned char>(filename, &buffer_size);

	if ( buffer != 0 ) {
		if ( n47::lzma::bufferIsLZMA(buffer, buffer_size) ) {
			read_bi5(data, buffer, buffer_size, point_value, bytes_read);
			// Reading in as bi5 failed lets double check its not binary
			// data in the buffer.
			if (data.size() == 0) {
				read_bin(data, buffer, buffer_size, point_value);
			}
		} else {
			read_bin(data, buffer, buffer_size, point_value);
			*bytes_read = buffer_size;
		}
		delete [] buffer;
}

}  // namespace n47
