-- phpMyAdmin SQL Dump
-- version 4.8.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jan 29, 2025 at 06:07 PM
-- Server version: 10.1.34-MariaDB
-- PHP Version: 7.2.8

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `shop3`
--

-- --------------------------------------------------------

--
-- Table structure for table `offer`
--

CREATE TABLE `offer` (
  `Original_Product` varchar(20) NOT NULL,
  `Discount_Product` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `offer`
--

INSERT INTO `offer` (`Original_Product`, `Discount_Product`) VALUES
('Hamam_Soap', 'Cinthol_Soap');

-- --------------------------------------------------------

--
-- Table structure for table `products`
--

CREATE TABLE `products` (
  `ID` int(10) NOT NULL,
  `Name` varchar(20) NOT NULL,
  `Category` varchar(20) NOT NULL,
  `Stock` int(10) NOT NULL,
  `Amount` int(10) NOT NULL,
  `High` int(10) NOT NULL,
  `Low` int(10) NOT NULL,
  `Side_low` int(10) NOT NULL,
  `Side_high` int(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `products`
--

INSERT INTO `products` (`ID`, `Name`, `Category`, `Stock`, `Amount`, `High`, `Low`, `Side_low`, `Side_high`) VALUES
(9, 'Cinthol_Soap', 'Bathing_Soap', 2, 10, 100000000, 0, 0, 100000000),
(10, 'Hamam_Soap', 'Bathing_Soap', 6, 10, 10000000, 0, 0, 10000000),
(11, 'Him_Face_Wash', 'Face_Wash', 6, 20, 10000000, 0, 0, 10000000),
(8, 'Maa_Juice', 'Soft_Drink', 10, 20, 10000000, 0, 0, 10000000),
(12, 'Mango', 'Fruits', 6, 20, 100000000, 0, 0, 1000000000),
(1, 'Mysore_Sandal_Soap_L', 'Bathing_Soap', -2, 40, 1000000, 20000, 31000, 10000000),
(2, 'Mysore_Sandal_Soap_S', 'Bathing_Soap', 4, 10, 50000, 0, 0, 30500),
(7, 'Patanjali_Dant_Kanti', 'Tooth_paste', 6, 30, 10000000, 0, 0, 10000000),
(5, 'Tide_Bar_Soap_L', 'Washing_Soap', 6, 40, 10000000, 50000, 30000, 100000000),
(6, 'Tide_Bar_Soap_S', 'Washing_Soap', 2, 10, 75000, 0, 0, 40000),
(3, 'ujala_liquid_L', 'Washing_Soap', 7, 40, 10000000, 30000, 40000, 10000000),
(4, 'ujala_liquid_S', 'Washing_Soap', 6, 10, 40000, 0, 0, 45000);

-- --------------------------------------------------------

--
-- Table structure for table `sold`
--

CREATE TABLE `sold` (
  `product` varchar(20) NOT NULL,
  `category` varchar(20) NOT NULL,
  `count` int(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `sold`
--

INSERT INTO `sold` (`product`, `category`, `count`) VALUES
('Cinthol_Soap', 'Bathing_Soap', 1),
('Hamam_Soap', 'Bathing_Soap', 3),
('Maa_Juice', 'Soft_Drink', 3),
('Mysore_Sandal_Soap_S', 'Bathing_Soap', 2),
('Tide_Bar_Soap_L', 'Washing_Soap', 2);

-- --------------------------------------------------------

--
-- Table structure for table `tally`
--

CREATE TABLE `tally` (
  `Time` timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  `mobile` text NOT NULL,
  `Total` int(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `tally`
--

INSERT INTO `tally` (`Time`, `mobile`, `Total`) VALUES
('2024-05-08 17:24:31.034075', '919941815173', 50),
('2025-01-19 09:55:04.082468', '919941815173', 40),
('2025-01-19 09:55:04.082468', '919941815173', 40),
('2025-01-19 09:55:04.082468', '919941815173', 200);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `products`
--
ALTER TABLE `products`
  ADD PRIMARY KEY (`Name`);

--
-- Indexes for table `sold`
--
ALTER TABLE `sold`
  ADD PRIMARY KEY (`product`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
