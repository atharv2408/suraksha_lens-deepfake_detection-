// src/utils/api.js
import axios from "axios";

const API = "http://127.0.0.1:8000";

export const analyzeImage = async (file) => {
  const form = new FormData();
  form.append("file", file);

  const res = await axios.post(`${API}/api/v1/analyze-image`, form, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return res.data;
};

export const generateComplaint = async (payload) => {
  const res = await axios.post(`${API}/api/v1/generate-complaint`, payload);
  return res.data; // { complaint_text: "..." }
};
