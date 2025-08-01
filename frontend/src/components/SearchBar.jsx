import React from "react";

const SearchBar = ({ query, onQueryChange, onSearch }) => {
  const handleInput = (e) => {
    onQueryChange(e.target.value);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      onSearch();
    }
  };

  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="Enter search query..."
        value={query}
        onChange={handleInput}
        onKeyPress={handleKeyPress}
      />
      <button onClick={onSearch}>Search</button>
    </div>
  );
};

export default SearchBar;
