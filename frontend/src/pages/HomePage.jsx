import React, { useState } from "react";
import SearchBar from "../components/SearchBar";
import ResultsList from "../components/ResultsList";
import * as api from "../services/api";

const HomePage = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    if (!query) return;
    try {
      const data = await api.searchDocuments(query);
      setResults(data.results || []);
    } catch (err) {
      console.error("Search error:", err);
    }
  };

  return (
    <div className="home-page">
      <h1>Digital Product Passport Semantic Search</h1>
      <SearchBar query={query} onQueryChange={setQuery} onSearch={handleSearch} />
      <ResultsList results={results} />
    </div>
  );
};

export default HomePage;
