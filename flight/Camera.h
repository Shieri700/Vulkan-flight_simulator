#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

class Camera
{
public:
	Camera(glm::vec3 positionIn, glm::vec3 upIn){
		position = positionIn;
		orUp = upIn;
		up = upIn;
		yaw = 0;
		pitch = 0;
		roll = 0.f;
		speed = 0.f;
		updateCameraVectors();
	}

	void updateCameraVectors() {
		front.x = cos(glm::radians(yaw - 90)) * cos(glm::radians(pitch));
		front.y = sin(glm::radians(pitch));
		front.z = sin(glm::radians(yaw - 90)) * cos(glm::radians(pitch));
		front = glm::normalize(front);
		glm::mat4 roll_mat = glm::rotate(glm::mat4(1.0f), glm::radians(roll), front);
		up = glm::mat3(roll_mat) * orUp;
	}

	void updatePos(float time) {
		position = position + front * speed * time;
	}

	glm::mat4 GetViewMatrix()
	{
		return glm::lookAt(position, position + front, up);
	}

	glm::vec3 getCentre() {
		return position + front;
	}

	void rolling(float amount) {
		roll += amount;
		if (roll >= 360) roll -= 360;
		else if (roll < 0) roll += 360;

		updateCameraVectors();
	}

	void pitching(float amount) {
		pitch += amount;
		if (pitch >= 360) pitch -= 360;
		else if (pitch < 0) pitch += 360;

		updateCameraVectors();
	}

	void yawing(float amount) {
		yaw += amount;
		if (yaw >= 360) yaw -= 360;
		else if (yaw < 0) yaw += 360;

		updateCameraVectors();
	}

	void speeding(float amount) {
		speed += amount;
	}

public:
	glm::vec3 position, up, orUp, front;
	float yaw, pitch, roll, speed;
};

