import React from 'react';
import Sidebar from './Sidebar';
import Header from './Header';

const Layout = ({ children }) => {
  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <Header />
        <div className="content-area animate-fade-in">
          {children}
        </div>
      </main>
    </div>
  );
};

export default Layout;
