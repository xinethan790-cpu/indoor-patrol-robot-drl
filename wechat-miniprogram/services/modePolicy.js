'use strict';

const DEFAULT_MODE = 'cloud';
const VALID_MODES = ['cloud', 'demo', 'local'];

function normalizeMode(value) {
  return VALID_MODES.includes(value) ? value : DEFAULT_MODE;
}

function resolveModeForPlatform(value, platform) {
  const requestedMode = normalizeMode(value);
  if (requestedMode === 'local' && platform !== 'devtools') {
    return {
      mode: 'cloud',
      requestedMode,
      downgraded: true,
      reason: 'local_mode_requires_devtools',
    };
  }
  return { mode: requestedMode, requestedMode, downgraded: false, reason: null };
}

function getPlatform(wxApi) {
  try {
    if (wxApi && typeof wxApi.getDeviceInfo === 'function') {
      return wxApi.getDeviceInfo().platform || 'unknown';
    }
    if (wxApi && typeof wxApi.getSystemInfoSync === 'function') {
      return wxApi.getSystemInfoSync().platform || 'unknown';
    }
  } catch (error) {
    return 'unknown';
  }
  return 'unknown';
}

module.exports = {
  DEFAULT_MODE,
  VALID_MODES,
  getPlatform,
  normalizeMode,
  resolveModeForPlatform,
};
