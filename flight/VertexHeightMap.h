#pragma once
#include <array>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

class VertexHeightMap
{
public:
	int height;
	glm::vec3 color;
	int width;
	int type;
	//glm::vec2 texCoord;

	VertexHeightMap() {}

	// describes at which rate to load data from memory throughout the vertices
	// all of per-vertex data is packed together in one array, so we're only going to have one binding
	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0; // index of the binding in the array of bindings
		bindingDescription.stride = sizeof(VertexHeightMap); // number of bytes from one entry to the next
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // Move to the next data entry after each vertex

		return bindingDescription;
	}

	// handle vertex input 
	// describes how to extract a vertex attribute from a chunk of vertex data originating from a binding description
	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
		// type
		attributeDescriptions[0].binding = 0; // which binding the per-vertex data comes
		attributeDescriptions[0].location = 0; // the location directive of the input in the vertex shader
		attributeDescriptions[0].format = VK_FORMAT_R16_SINT; // type of data for the attribute
		attributeDescriptions[0].offset = offsetof(VertexHeightMap, type); // number of bytes since the start of the per-vertex data to read from

		// colour
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(VertexHeightMap, color);

		// texture coord
		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R16_SINT;
		attributeDescriptions[2].offset = offsetof(VertexHeightMap, width);

		return attributeDescriptions;
	}
};

