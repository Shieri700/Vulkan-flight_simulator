#pragma once
#include <vector>
#include "Vertex.h"
class HeightMap
{
public:
	HeightMap() {}
	~HeightMap() { heightdata.clear(); }
	void loadheights(std::vector<std::vector<int>> heightsIn, uint32_t patchSize);

	std::vector<uint32_t> loadIndices();
	float getHeight(uint32_t x, uint32_t z);

public:
	uint32_t scale;
	uint32_t width;
	std::vector<int> heightdata;
};

