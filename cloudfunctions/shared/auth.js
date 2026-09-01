'use strict';

const crypto = require('node:crypto');

const {
  MAX_BODY_BYTES,
  MAX_CLOCK_SKEW_SECONDS,
  SIGNATURE_VERSION,
} = require('./config');
const { badRequest, unauthorized } = require('./errors');

const DEVICE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const NONCE_PATTERN = /^[A-Za-z0-9._~-]{16,128}$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const TIMESTAMP_PATTERN = /^\d{10,12}$/;

function normalizedHeaders(event) {
  const candidates = [event && event.headers, event && event.header];
  const output = {};
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) continue;
    for (const [key, value] of Object.entries(candidate)) {
      output[String(key).toLowerCase()] = Array.isArray(value) ? value[0] : String(value);
    }
  }
  return output;
}

function extractMethod(event) {
  const method = event && (
    event.httpMethod
    || event.method
    || (event.requestContext && event.requestContext.http && event.requestContext.http.method)
  );
  return method ? String(method).toUpperCase() : 'POST';
}

function extractRawBody(event) {
  if (!event || typeof event !== 'object') throw badRequest('INVALID_REQUEST', 'request event is missing');
  let raw = event.rawBody !== undefined ? event.rawBody : event.body;
  if (Buffer.isBuffer(raw)) raw = raw.toString('utf8');
  if (typeof raw !== 'string') {
    throw badRequest('RAW_BODY_REQUIRED', 'HTTP gateway must pass the original JSON body as text');
  }
  if (event.isBase64Encoded === true) {
    try {
      raw = Buffer.from(raw, 'base64').toString('utf8');
    } catch (error) {
      throw badRequest('INVALID_BODY_ENCODING', 'request body is not valid base64');
    }
  }
  const byteLength = Buffer.byteLength(raw, 'utf8');
  if (byteLength === 0) throw badRequest('EMPTY_BODY', 'request body cannot be empty');
  if (byteLength > MAX_BODY_BYTES) {
    const error = badRequest('BODY_TOO_LARGE', `request body exceeds ${MAX_BODY_BYTES} bytes`);
    error.statusCode = 413;
    throw error;
  }
  return raw;
}

function safeHexEqual(left, right) {
  if (!SHA256_PATTERN.test(left) || !SHA256_PATTERN.test(right)) return false;
  return crypto.timingSafeEqual(Buffer.from(left, 'hex'), Buffer.from(right, 'hex'));
}

function authenticateUpload(event, config, nowEpochMs) {
  if (extractMethod(event) !== 'POST') {
    const error = badRequest('METHOD_NOT_ALLOWED', 'only POST is supported');
    error.statusCode = 405;
    throw error;
  }
  if (!config.uploadSecret || Buffer.byteLength(config.uploadSecret, 'utf8') < 32) {
    throw new Error('PATROL_RELAY_UPLOAD_SECRET must contain at least 32 UTF-8 bytes');
  }

  const headers = normalizedHeaders(event);
  const signatureVersion = headers['x-patrol-signature-version'];
  const deviceId = headers['x-patrol-device-id'];
  const timestamp = headers['x-patrol-timestamp'];
  const nonce = headers['x-patrol-nonce'];
  const suppliedBodySha256 = headers['x-patrol-body-sha256'];
  const suppliedSignature = headers['x-patrol-signature'];
  const contentType = headers['content-type'];

  if (!contentType || !contentType.toLowerCase().startsWith('application/json')) {
    const error = badRequest('UNSUPPORTED_MEDIA_TYPE', 'content type must be application/json');
    error.statusCode = 415;
    throw error;
  }

  if (signatureVersion !== SIGNATURE_VERSION) {
    throw unauthorized('SIGNATURE_VERSION_MISMATCH', 'unsupported signature version');
  }
  if (!deviceId || !DEVICE_ID_PATTERN.test(deviceId)) {
    throw unauthorized('INVALID_DEVICE_ID', 'device id is missing or invalid');
  }
  if (config.deviceId && deviceId !== config.deviceId) {
    throw unauthorized('DEVICE_NOT_ALLOWED', 'device is not allowed to upload');
  }
  if (!timestamp || !TIMESTAMP_PATTERN.test(timestamp)) {
    throw unauthorized('INVALID_TIMESTAMP', 'timestamp must be Unix epoch seconds');
  }
  const timestampSeconds = Number(timestamp);
  const nowSeconds = Math.floor(nowEpochMs / 1000);
  if (!Number.isSafeInteger(timestampSeconds)
      || Math.abs(timestampSeconds - nowSeconds) > MAX_CLOCK_SKEW_SECONDS) {
    throw unauthorized('STALE_TIMESTAMP', 'request timestamp is outside the accepted clock window');
  }
  if (!nonce || !NONCE_PATTERN.test(nonce)) {
    throw unauthorized('INVALID_NONCE', 'nonce is missing or invalid');
  }
  if (!suppliedBodySha256 || !SHA256_PATTERN.test(suppliedBodySha256)) {
    throw unauthorized('INVALID_BODY_DIGEST', 'body digest is missing or invalid');
  }
  if (!suppliedSignature || !SHA256_PATTERN.test(suppliedSignature)) {
    throw unauthorized('INVALID_SIGNATURE', 'signature is missing or invalid');
  }

  const rawBody = extractRawBody(event);
  const actualBodySha256 = crypto.createHash('sha256').update(rawBody, 'utf8').digest('hex');
  if (!safeHexEqual(suppliedBodySha256, actualBodySha256)) {
    throw unauthorized('BODY_DIGEST_MISMATCH', 'body digest does not match the request body');
  }
  const canonical = `${deviceId}\n${timestamp}\n${nonce}\n${actualBodySha256}`;
  const expectedSignature = crypto
    .createHmac('sha256', config.uploadSecret)
    .update(canonical, 'utf8')
    .digest('hex');
  if (!safeHexEqual(suppliedSignature, expectedSignature)) {
    throw unauthorized('SIGNATURE_MISMATCH', 'request signature is invalid');
  }

  return {
    bodySha256: actualBodySha256,
    deviceId,
    nonce,
    rawBody,
    requestTimestampEpochS: timestampSeconds,
  };
}

module.exports = {
  DEVICE_ID_PATTERN,
  NONCE_PATTERN,
  SHA256_PATTERN,
  authenticateUpload,
  extractRawBody,
  normalizedHeaders,
};
