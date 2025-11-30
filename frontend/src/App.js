import React, { useState } from 'react';
import styled from 'styled-components';
import ImageUploader from './components/ImageUploader';
import ComparisonGrid from './components/ComparisonGrid';
import LoadingSpinner from './components/LoadingSpinner';
import Header from './components/Header';
import './App.css';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const MainContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  background: white;
  border-radius: 15px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.1);
  overflow: hidden;
`;

const ContentArea = styled.div`
  padding: 30px;
`;

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/upload-image/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setResults(data);
      } else {
        throw new Error('Server returned unsuccessful response');
      }

    } catch (err) {
      console.error('Upload error:', err);
      setError(`Failed to process image: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
  };

  return (
    <AppContainer>
      <MainContent>
        <Header />
        <ContentArea>
          {!results && !loading && (
            <ImageUploader 
              onImageUpload={handleImageUpload} 
              error={error}
            />
          )}
          
          {loading && <LoadingSpinner />}
          
          {results && (
            <ComparisonGrid 
              results={results} 
              onReset={handleReset}
            />
          )}
          
          {error && !loading && (
            <div style={{ 
              color: 'red', 
              textAlign: 'center', 
              padding: '20px',
              background: '#ffe6e6',
              borderRadius: '10px',
              margin: '20px 0'
            }}>
              <h3>Error</h3>
              <p>{error}</p>
              <button 
                onClick={handleReset}
                style={{
                  background: '#007bff',
                  color: 'white',
                  border: 'none',
                  padding: '10px 20px',
                  borderRadius: '5px',
                  cursor: 'pointer',
                  marginTop: '10px'
                }}
              >
                Try Again
              </button>
            </div>
          )}
        </ContentArea>
      </MainContent>
    </AppContainer>
  );
}

export default App;