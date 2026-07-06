import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import StudyRooms from './pages/StudyRooms';
import RoomDetail from './pages/RoomDetail';
import Planner from './pages/Planner';
import Leaderboard from './pages/Leaderboard';
import Projects from './pages/Projects';
import ProjectDetail from './pages/ProjectDetail';
import MainLayout from './components/layout/MainLayout';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/login" element={<Login onLogin={() => setIsAuthenticated(true)} />} />
        
        {/* Protected Routes */}
        <Route element={isAuthenticated ? <MainLayout /> : <Navigate to="/login" />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/rooms" element={<StudyRooms />} />
          <Route path="/rooms/:roomId" element={<RoomDetail />} />
          <Route path="/planner" element={<Planner />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:id" element={<ProjectDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
