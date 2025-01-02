/**
 * Saves a value to localStorage under the given key.
 * @param {string} key - The localStorage key name.
 * @param {*} value - The value to store (will be coerced to string).
 */
export const setLocal = (key, value) => {
  try {
    localStorage?.setItem(key, String(value));
  } catch (err) {
    console.warn(`Failed to set "${key}" in localStorage:`, err);
  }
};

/**
 * Retrieves a value from localStorage by key.
 * @param {string} key - The localStorage key name.
 * @param {*} [defaultValue] - A default value if no item is found or on error.
 * @returns {*} The retrieved string or the defaultValue if missing/error.
 */
export const getLocal = (key, defaultValue = null) => {
  try {
    const item = localStorage?.getItem(key);
    return item !== null ? item : defaultValue;
  } catch (err) {
    console.warn(`Failed to get "${key}" from localStorage:`, err);
    return defaultValue;
  }
};

/**
 * Removes an item from localStorage by key.
 * @param {string} key - The localStorage key name.
 */
export const removeLocal = (key) => {
  try {
    localStorage?.removeItem(key);
  } catch (err) {
    console.warn(`Failed to remove "${key}" from localStorage:`, err);
  }
};
