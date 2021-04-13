#include "HeightMap.h"

void HeightMap::loadheights(std::vector<std::vector<int>> heightsIn, uint32_t patchSize) {
	width = heightsIn.size();

	scale = patchSize / width;

	for (int h = 0; h < heightsIn.size(); h++) {
		for (int w = 0; w < heightsIn[h].size(); w++) {
			heightdata.push_back(-heightsIn[h][w]);
		}
	}
}

std::vector<uint32_t> HeightMap::loadIndices() {
	std::vector<uint32_t> indicesData;
	for (uint32_t h = 0; h < width - 1; h++) {
		for (uint32_t w = 0; w < width - 1; w++) {

			indicesData.push_back(h * width + w + 1);
			indicesData.push_back(h * width + w);
			indicesData.push_back((h + 1) * width + w);

			indicesData.push_back((h + 1) * width + w);
			indicesData.push_back((h + 1) * width + w + 1);
			indicesData.push_back(h * width + w + 1);

		}
	}
	return indicesData;
}

float HeightMap::getHeight(uint32_t x, uint32_t z){
	return heightdata[x * width + z];
}
