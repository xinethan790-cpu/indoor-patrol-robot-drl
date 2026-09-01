'use strict';

class PatrolCloudError extends Error {
  constructor(code, statusCode, message) {
    super(message);
    this.name = 'PatrolCloudError';
    this.code = code;
    this.statusCode = statusCode;
  }
}

function badRequest(code, message) {
  return new PatrolCloudError(code, 400, message);
}

function unauthorized(code, message) {
  return new PatrolCloudError(code, 401, message);
}

function conflict(code, message) {
  return new PatrolCloudError(code, 409, message);
}

module.exports = {
  PatrolCloudError,
  badRequest,
  conflict,
  unauthorized,
};
