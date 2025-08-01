import React from "react";

const ResultsList = ({ results }) => {
  if (results.length === 0) {
    return <div className="results-list">No results found.</div>;
  }

  return (
    <div className="results-list">
      {results.map((res, index) => (
        <div key={index} className="result-item">
          <h3>{res.document_name}</h3>
          <p>{res.snippet}</p>
          <small>Score: {res.score.toFixed(4)}</small>
        </div>
      ))}
    </div>
  );
};

export default ResultsList;
